import sys
import torch.multiprocessing as tmp
import subprocess
import time
import os
import queue
import hivemind # type: ignore
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.deterministic = False # NOTE deactivated due to using set_all_seeds()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True # NOTE added for H100 compatibility

import math
import copy
import random
import gc
import pickle
import argparse
import wandb
from datetime import datetime
import signal
import threading
import pathlib
from typing import Dict, Any, Tuple, List

shutdown_event = threading.Event()

from mnist_bert_fl_ml_modules import SimpleCNN, ModernBERTClassifier, get_mnist_data_loaders, get_mnist_data_loaders_dirichlet, get_text_data_loaders, get_text_data_loaders_dirichlet, train_num_mini_batches_manually, evaluate, set_all_seeds



# -------------------- Parameters --------------------

ML_TASK = "mnist"

KD_NUM_DATASET_SIGHTINGS = 10                                               # NOTE usually use value 10; how often each peer sees his entire training dataloader's batches while using KD

TESTING = False

ENTITY = "P2PFLPersonalized"
PROJECT = "p2p_fl"

DEVICE = "cuda"                                                             # NOTE "cpu" or "cuda"
ADMIN = False                                                               # NOTE admin permissions (e.g., to kill zombie processes)

NUM_PEERS = 64
PEERS_PER_CORE = 1                                                          # NOTE set peers_per_core=2 when num_peers=64 while --ntasks=1 & --cpus-per-task=32

LOCAL_UPDATE_PARTICIPATION_PROB = 100
AGGREGATION_PARTICIPATION_PROB = 100
PEER_DROPOUT_LIKELIHOOD = 0

ITERATIONS_MECHANISM_APPROACH = 0                                           # NOTE set to 0 for "mini batches per iteration" or to 1 for "iterations per dataset"
ITERATIONS_MECHANISM_VALUE = 2                                              # NOTE if for instance 71 mini-batches per client then approach 0 in combination with value 25 equals approach 1 in combination with value 3
MAX_ITERATIONS = 500                                                        # NOTE int(TOTAL_BATCHES / MINI_BATCHES_PER_ITERATION) * 20

LEARNING_RATE = 0.1
MOMENTUM = 0.9

TESTING_FREQUENCY = 5                                                       # NOTE set to 5 s.t. peers do model testing every iteration i for which 'i % 5 == 0'
CONVERGENCE_PATIENCE = MAX_ITERATIONS * TESTING_FREQUENCY                   # NOTE consider the TESTING_FREQUENCY; int(MAX_ITERATIONS / 20) * 20 * TESTING_FREQUENCY
CONVERGENCE_THRESHOLD = 1e-3

DISPATCHER_PATIENCE = 30 * 60

ACCURACY_THRESHOLD = 0.0      # Minimum accuracy improvement to accept update (percentage points)
LOSS_THRESHOLD = 0.0          # Minimum loss reduction to accept update
ENABLE_VALIDATION = True

# ==================== UTILITY VALIDATOR ====================

class UtilityValidator:
    """
    Core validation component for ModelChain protocol.
    
    Tests if an incoming model update improves performance on the peer's
    local validation set. This is the "proof of utility" - analogous to
    proof-of-work in blockchain.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with thresholds from config."""
        self.accuracy_threshold = config.get('accuracy_threshold', 0.0)
        self.loss_threshold = config.get('loss_threshold', 0.0)
        
    def validate(
        self, 
        peer_model: nn.Module, 
        incoming_state_dict: Dict[str, torch.Tensor], 
        device: torch.device, 
        validation_loader, 
        peer_id: int,
        source_peer_id: int = -1
    ) -> Dict[str, Any]:
        """
        Validate if incoming model improves performance on local validation set.
        
        Returns:
            dict: Validation result containing:
                - valid (bool): Whether update should be accepted
                - improvement (dict): accuracy_delta and loss_delta
                - baseline_metrics (dict): Current model performance
                - incoming_metrics (dict): Incoming model performance
                - decision_reason (str): Human-readable explanation
        """
        try:
            # Step 1: Evaluate current model (baseline)
            baseline_metrics = self._evaluate_model(peer_model, device, validation_loader)
            
            # Step 2: Create temporary model for testing incoming weights
            temp_model = copy.deepcopy(peer_model)
            
            # Step 3: Try to load incoming state dict
            try:
                temp_model.load_state_dict(incoming_state_dict, strict=False)
            except RuntimeError as e:
                del temp_model
                return {
                    'valid': False,
                    'improvement': {'accuracy_delta': -999.0, 'loss_delta': -999.0},
                    'baseline_metrics': baseline_metrics,
                    'incoming_metrics': {'accuracy': 0.0, 'loss': float('inf'), 'samples_evaluated': 0},
                    'decision_reason': f"[Peer {peer_id}] REJECT from Peer {source_peer_id}: Architecture mismatch"
                }
            
            # Step 4: Evaluate incoming model
            incoming_metrics = self._evaluate_model(temp_model, device, validation_loader)
            
            # Step 5: Compute improvement
            improvement = {
                'accuracy_delta': incoming_metrics['accuracy'] - baseline_metrics['accuracy'],
                'loss_delta': baseline_metrics['loss'] - incoming_metrics['loss']  # Positive = improvement
            }
            
            # Step 6: Make decision based on thresholds
            is_beneficial = (
                improvement['accuracy_delta'] > self.accuracy_threshold or
                improvement['loss_delta'] > self.loss_threshold
            )
            
            decision_reason = self._get_decision_reason(improvement, is_beneficial, peer_id, source_peer_id)
            
            # Step 7: Cleanup
            del temp_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'valid': is_beneficial,
                'improvement': improvement,
                'baseline_metrics': baseline_metrics,
                'incoming_metrics': incoming_metrics,
                'decision_reason': decision_reason
            }
            
        except Exception as e:
            return {
                'valid': False,
                'improvement': {'accuracy_delta': -999.0, 'loss_delta': -999.0},
                'baseline_metrics': {'accuracy': 0.0, 'loss': float('inf'), 'samples_evaluated': 0},
                'incoming_metrics': {'accuracy': 0.0, 'loss': float('inf'), 'samples_evaluated': 0},
                'decision_reason': f"[Peer {peer_id}] REJECT from Peer {source_peer_id}: Validation error - {str(e)[:50]}"
            }
    
    def _evaluate_model(self, model: nn.Module, device: torch.device, validation_loader) -> Dict[str, float]:
        """Evaluate model on validation dataset."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in validation_loader:
                try:
                    target = target.to(device)
                    if isinstance(data, dict):
                        input_ids = data['input_ids'].to(device)
                        attention_mask = data['attention_mask'].to(device)
                        output = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        data = data.to(device)
                        output = model(data)
                    
                    loss = torch.nn.functional.cross_entropy(output, target, reduction='mean')
                    total_loss += loss.item()
                    num_batches += 1
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                except Exception as e:
                    continue
        
        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        avg_loss = (total_loss / num_batches) if num_batches > 0 else float('inf')
        return {'accuracy': accuracy, 'loss': avg_loss, 'samples_evaluated': total}
    
    def _get_decision_reason(self, improvement: Dict[str, float], is_beneficial: bool, peer_id: int, source_peer_id: int) -> str:
        if is_beneficial:
            return f"[Peer {peer_id}] ACCEPT from Peer {source_peer_id}: Accuracy {improvement['accuracy_delta']:+.2f}%, Loss {improvement['loss_delta']:+.4f}"
        else:
            return f"[Peer {peer_id}] REJECT from Peer {source_peer_id}: Accuracy {improvement['accuracy_delta']:+.2f}% (threshold: {self.accuracy_threshold}%)"



# ==================== MODEL LEDGER ====================

class ModelLedger:
    """
    Blockchain-like ledger for tracking model evolution history.
    Records all integration and rejection decisions.
    """
    
    def __init__(self, peer_id: int):
        self.peer_id = peer_id
        self.chain: List[Dict] = []
        self.rejected_cache: Dict[str, Dict] = {}
        
    def add_integration(self, iteration: int, sender_id: int, validation_result: Dict[str, Any]) -> None:
        """Add an accepted update to the ledger chain."""
        block = {
            'block_number': len(self.chain),
            'timestamp': time.time(),
            'iteration': iteration,
            'sender_id': sender_id,
            'validation_result': validation_result,
            'improvement': validation_result.get('improvement', {})
        }
        self.chain.append(block)
        
    def add_rejection(self, iteration: int, sender_id: int, validation_result: Dict[str, Any]) -> None:
        """Add a rejected update to the rejection cache."""
        key = f"{iteration}_{sender_id}"
        self.rejected_cache[key] = {
            'timestamp': time.time(),
            'iteration': iteration,
            'sender_id': sender_id,
            'reason': validation_result.get('decision_reason', 'Unknown')
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute aggregate statistics from the ledger."""
        total_integrations = len(self.chain)
        total_rejections = len(self.rejected_cache)
        
        if total_integrations > 0:
            avg_accuracy_gain = sum(block['improvement'].get('accuracy_delta', 0.0) for block in self.chain) / total_integrations
            avg_loss_reduction = sum(block['improvement'].get('loss_delta', 0.0) for block in self.chain) / total_integrations
        else:
            avg_accuracy_gain = 0.0
            avg_loss_reduction = 0.0
        
        total_decisions = total_integrations + total_rejections
        acceptance_rate = (total_integrations / total_decisions) if total_decisions > 0 else 0.0
        
        return {
            'total_integrations': total_integrations,
            'total_rejections': total_rejections,
            'acceptance_rate': acceptance_rate,
            'avg_accuracy_gain': avg_accuracy_gain,
            'avg_loss_reduction': avg_loss_reduction
        }

# ==================== HELPER FUNCTION ====================

def create_validation_components(peer_id: int, accuracy_threshold: float = 0.0, loss_threshold: float = 0.0) -> Tuple[UtilityValidator, ModelLedger]:
    """Factory function to create validator and ledger together."""
    config = {'accuracy_threshold': accuracy_threshold, 'loss_threshold': loss_threshold}
    validator = UtilityValidator(config)
    ledger = ModelLedger(peer_id)
    return validator, ledger


# -------------------- Utils --------------------

# Shared memory load measurement regarding python multiprocessing and torch
def _dev_shm_usage_bytes(select_prefixes=("psm_", "torch_")):
    total = 0
    shm_path = pathlib.Path("/dev/shm")
    if not shm_path.exists(): # NOTE not a Linux box or no tmpfs mounted
        return 0
    for f in shm_path.iterdir():
        try:
            if not select_prefixes or f.name.startswith(select_prefixes):
                total += f.stat().st_size
        except FileNotFoundError: # NOTE the file disappeared between listdir and stat – ignore
            pass
    return total

# Shared memory load measurement independent of library prefix
def _dev_shm_total_usage_bytes():
    total = 0
    shm_path = pathlib.Path("/dev/shm")
    if not shm_path.exists():
        return 0
    for f in shm_path.iterdir():
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total

# Shared memory logging
def _log_shared_memory(prefix: str, shared_dict, step=None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mb = _dev_shm_usage_bytes() / 1024 / 1024
    n_entries = len(shared_dict)
    print(f"[{ts}][{prefix}] /dev/shm ≈ {mb:8.1f} MB | shared_model_dict entries: {n_entries}", flush=True)
    log_dict = {
        "shm_intermediate_MB": mb,
        "shm_shared_dict_entries": n_entries
    }
    try:
        wandb.log(log_dict)
    except Exception as e:
        print(f"[{ts}][{prefix}] WARNING: wandb.log failed: {e}", flush=True)

# Constant shared memory watcher
def start_shm_watcher(shared_dict,
                      prefix: str,
                      interval: int = 30) -> threading.Thread:
    def _worker():
        while not shutdown_event.is_set():
            _log_shared_memory(prefix, shared_dict)
            time.sleep(interval)
        _log_shared_memory(prefix + "-final", shared_dict) # NOTE one last measurement on exit
    t = threading.Thread(target=_worker, name=f"{prefix}-shm‑watcher", daemon=True)
    t.start()
    return t

# Signal handler for cleanup
def signal_handler(signum, frame):
    print(f"Received signal {signum}, shutting down gracefully...", flush=True)
    shutdown_event.set()
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Generate a unique and informative wandb run name
def generate_unique_run_name(split_type, entity, project, seed, ml_task, num_peers, mini_batches_per_iteration, aggregation_participation_prob, local_update_participation_prob, peer_dropout_likelihood):
    api = wandb.Api()
    base_name = f"ar{num_peers}{ml_task}_b{mini_batches_per_iteration}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}__s{seed}"
    if local_update_participation_prob < 100:
        base_name = f"ar{num_peers}{ml_task}_b{mini_batches_per_iteration}_u{local_update_participation_prob}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}__s{seed}"
    if split_type == "dirichlet":
        base_name += "_niid"
    existing_runs = api.runs(f"{entity}/{project}", filters={"display_name": {"$regex": f"^{base_name}(_\\d+)?$"}})
    existing_names = {run.name for run in existing_runs}
    if base_name not in existing_names:
        return base_name
    else:
        suffix = 1
        while f"{base_name}_{suffix}" in existing_names:
            suffix += 1
        return f"{base_name}_{suffix}"

# Wait until all num_peers_target amongst num_peers have set a DHT flag for a specific prefix which optimally includes the iteration number
def wait_for_all(dht, prefix, num_peers, num_peers_target, delay=0.05, max_delays=200):
    delay_counter = 0
    while True:
        ready_peers = 0
        written_members = []
        for peer in range(num_peers):
            key = f"{prefix}_{peer}"
            if dht.get(key) is not None:
                ready_peers += 1
                written_members.append(peer)
        if ready_peers == num_peers_target or delay_counter == max_delays:
            return written_members
        time.sleep(delay)
        delay_counter += 1

# Wait until all specified peers have set a DHT flag for a specific prefix which optimally includes the iteration number
def wait_for_peers(dht, prefix, peers, delay=0.05, max_delays=200):
    delay_counter = 0
    while True:
        all_set = all(dht.get(f"{prefix}_{peer}", latest=True) is not None for peer in peers)
        if all_set:
            return True
        if delay_counter == max_delays:
            return False
        time.sleep(delay)
        delay_counter += 1

# Announce participation and count number of participating peers
def count_num_participating_peers(peer_id, iteration, num_peers, dht, wait_time=5):
    participation_indicator_key = f"pik_{iteration}"
    dht.store(f"{participation_indicator_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    time.sleep(wait_time)
    participating_peers = []
    for peer in range(num_peers):
        key = f"{participation_indicator_key}_{peer}"
        if dht.get(key) is not None:
            participating_peers.append(peer)
    return len(participating_peers), sorted(participating_peers)

# Communicate model parameters and momentum vectors within groups through writing to and reading from shared group dicts
def communicate_models(testing, device, peer_id, num_peers, num_participating_peers, model, momentum_vector, shared_model_dict, dht, iteration, round, communicated_bytes,
                       validation_loader=None, validator=None, ledger=None, enable_validation=False):
    
    # Prepare data to be communicated
    iter_key = f"{iteration}-{round}"
    if isinstance(model, ModernBERTClassifier):
        state_dict_to_send = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        momentum_to_send = [mv.cpu().clone() for p, mv in zip(model.parameters(), momentum_vector) if p.requires_grad]
    else:
        state_dict_to_send = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        momentum_to_send = [mv.cpu().clone() for mv in momentum_vector]

    # Store model & momentum vector in group dict and inform other peers via dht and wait for all participating peers to write

    shared_model_dict[(iter_key, peer_id)] = {"model_state_dict": state_dict_to_send, "momentum_vector": momentum_to_send}
    barrier_key = f"cm1_{iter_key}"
    dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    if testing:
        print(f"Peer {peer_id} cm1 in iteration {iteration}-{round}.", flush=True)
    written_members = wait_for_all(dht, barrier_key, num_peers, num_participating_peers)

    # 1v5: initialize sum with own model/momentum
    if isinstance(model, ModernBERTClassifier):
        integrated_state_dict = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        integrated_momentum = [mv.clone() for p, mv in zip(model.parameters(), momentum_vector) if p.requires_grad]
    else:
        integrated_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        integrated_momentum = [mv.clone() for mv in momentum_vector]
    integrated_count = 1
    accepted_peers = [peer_id]
    rejected_peers = []

    # 2v5: stream, sum and discard models from other peers
    for source_peer in written_members:
        if source_peer == peer_id: continue
        try:
            entry = shared_model_dict[(iter_key, source_peer)]
            incoming_model_gpu = {k: v.to(device) for k, v in entry["model_state_dict"].items()}
            incoming_momentum_gpu = [m.to(device) for m in entry["momentum_vector"]]

            should_integrate = True
            if enable_validation and validator is not None and validation_loader is not None:
                validation_result = validator.validate(
                    peer_model=model,
                    incoming_state_dict=incoming_model_gpu,
                    device=device,
                    validation_loader=validation_loader,
                    peer_id=peer_id,
                    source_peer_id=source_peer
                )
                should_integrate = validation_result['valid']
                print(validation_result['decision_reason'], flush=True)

                 # Record in ledger
                if ledger is not None:
                    if should_integrate:
                        ledger.add_integration(iteration, source_peer, validation_result)
                    else:
                        ledger.add_rejection(iteration, source_peer, validation_result)
            
            if should_integrate:
                for key in integrated_state_dict:
                    if key in incoming_model_gpu:
                        integrated_state_dict[key].add_(incoming_model_gpu[key])
                for i in range(len(integrated_momentum)):
                    integrated_momentum[i].add_(incoming_momentum_gpu[i])
                integrated_count += 1
                accepted_peers.append(source_peer)
            else:
                rejected_peers.append(source_peer)

            del incoming_model_gpu, incoming_momentum_gpu
            
        except KeyError:
            print(f"[{datetime.now()}] WARNING: Peer {peer_id} could not load model/momentum from peer {peer}.")

    # 3v5: Average only the integrated models
    if integrated_count > 0:
        for key in integrated_state_dict:
            integrated_state_dict[key].div_(integrated_count)
        for i in range(len(integrated_momentum)):
            integrated_momentum[i].div_(integrated_count)

    # Log integration statistics
    print(f"[Peer {peer_id}] Integrated {integrated_count}/{len(written_members)} models. "
          f"Accepted: {accepted_peers}, Rejected: {rejected_peers}", flush=True)
    
    # 4v5: track communication traffic
    communicated_bytes += (sum(p.numel() * p.element_size() for p in integrated_state_dict.values()) + 
                           sum(m.numel() * m.element_size() for m in integrated_momentum)) * (integrated_count -1)

    # 5v5: cleanup if we are the coordinator
    barrier_key_cleanup = f"cm2_{iter_key}"
    dht.store(f"{barrier_key_cleanup}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    if peer_id == min(written_members): # Coordinator cleans up the shared dict
        wait_for_peers(dht, barrier_key_cleanup, written_members)
        for peer in written_members:
            try:
                if (iter_key, peer) in shared_model_dict:
                    del shared_model_dict[(iter_key, peer)]
            except KeyError:
                pass
        torch.cuda.empty_cache()
        gc.collect()        
    return integrated_state_dict, integrated_momentum, len(written_members), communicated_bytes


def peer_process(seed, split_type, dirichlet_alpha, ml_task, peer_id, bootstrap_address, task_queue, result_queue, num_peers, peers_per_core, learning_rate, momentum, shared_model_dict, valid_cores, device, model, momentum_vector=None, next_batch_idx=None
                 , enable_validation=False, accuracy_threshold=0.2, loss_threshold=0.2):
    try:
        """ Each peer runs persistently and waits for tasks to execute """

        # Set random seeds for reproducibility
        set_all_seeds(seed)

        # Bind groups of peers_per_core peers to the same CPU core (e.g. set peers_per_core=2 when num_peers=64 while --ntasks=1 & --cpus-per-task=32)
        vcpu_id = valid_cores[(peer_id // peers_per_core) % len(valid_cores)]
        os.sched_setaffinity(0, {vcpu_id})

        # Initialize DHT instance
        dht = hivemind.DHT(initial_peers=bootstrap_address, start=True)
        print(f"Peer {peer_id} started on vCPU: {list(os.sched_getaffinity(0))} | Process ID: {os.getpid()}")

        # Testing logs for debugging
        if TESTING:
            log_group_id = (peer_id // 10) * 10
            log_path = f"ar{num_peers}-{ml_task}-testing_peers{log_group_id}-{log_group_id+9}.log"
            sys.stdout = open(log_path, "a", buffering=1)
            sys.stderr = sys.stdout

        # Initialize model data
        if device == "cuda":
            assert torch.cuda.is_available(), f"[{datetime.now()}] WARNING: Peer {peer_id} requested CUDA but not available."
            torch.cuda.set_device(peer_id % torch.cuda.device_count())
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device) # We will need to have different models per peer process, So maybe we should pass in
                                 #  different models per peer process from the main process
        if ml_task == "news": # That leaves with having to load different datasets per peer process
            if split_type == "dirichlet":
                train_loader, validation_loader, test_loader = get_text_data_loaders_dirichlet(partition=peer_id, num_partitions=num_peers, seed=seed, alpha=dirichlet_alpha)
            else:
                train_loader, validation_loader, test_loader = get_text_data_loaders(partition=peer_id, num_partitions=num_peers, seed=seed)
        elif ml_task == "mnist":
            if split_type == "dirichlet":
                train_loader, validation_loader, test_loader = get_mnist_data_loaders_dirichlet(partition=peer_id, num_partitions=num_peers, seed=seed, alpha=dirichlet_alpha)
            else:
                train_loader, validation_loader, test_loader = get_mnist_data_loaders(partition=peer_id, num_partitions=num_peers, seed=seed)

        # Added things for ModelChain validation
        validator, ledger = None, None
        if enable_validation:
            validator, ledger = create_validation_components(
                peer_id=peer_id,
                accuracy_threshold=accuracy_threshold,
                loss_threshold=loss_threshold
            )
            print(f"[Peer {peer_id}] ModelChain validation ENABLED", flush=True)
        else:
            print(f"[Peer {peer_id}] ModelChain validation DISABLED", flush=True)
        
        num_batches = len(train_loader)
        if momentum_vector is None:
            momentum_vector = [torch.zeros_like(p, device=device) for p in model.parameters()]
        if next_batch_idx is None:
            next_batch_idx = 0
        print(f"Peer {peer_id} using {device} as device for model training.")

        while True:
            if shutdown_event.is_set(): # NOTE added cleanup best practice
                print(f"Peer {peer_id} received shutdown signal, exiting loop.")
                break

            try:
                # Get a task from the task queue and skip it and put it back into the queue if it is dedicated for somebody else
                queue_item = task_queue.get(timeout=0.1)
                try:
                    target_peer, (task, iteration, task_value) = queue_item
                except (ValueError, TypeError) as e:
                    print(f"Peer {peer_id} received a malformed task: {queue_item}. Error: {e}. Skipping.", flush=True)
                    continue
                if target_peer not in (peer_id, "All"):
                    task_queue.put(queue_item)
                    continue
                print(f"Peer {peer_id} got task: {task} with value {task_value}...")



                # -------------------- Client Update --------------------

                if task == "update":

                    # Train model on the peer's local and distinct data partition (core FL principle)
                    training_message = ""
                    training_loss = 0
                    start_time_update = time.time()
                    training_loss, training_message, next_batch_idx, momentum_vector, _ = train_num_mini_batches_manually( # NOTE increments the next batch index by the number of mini batches we just passed through the model
                        model=model, 
                        device=device, 
                        train_loader=train_loader, 
                        learning_rate=learning_rate,
                        momentum=momentum,
                        momentum_vector=momentum_vector, 
                        peer_id=peer_id,
                        num_mini_batches=task_value,
                        next_batch_idx=next_batch_idx
                    )
                    training_duration = time.time() - start_time_update
                    remaining_batches = num_batches - next_batch_idx

                    # Notify dispatcher
                    if iteration % 5 == 0:
                        training_message += f" Torch threads: {torch.get_num_threads()} (PID: {os.getpid()}). Running on CPU(s): {list(os.sched_getaffinity(0))}."
                    result_queue.put((3, iteration, peer_id, training_loss, training_message, (remaining_batches, training_duration)))

                # -------------------- Model Aggregation --------------------

                elif task == "aggregate":

                    # Unpack task value tuple
                    start_time_aggregation = time.time()
                    num_participating_peers, do_testing = task_value

                    # To align to real-world scenarios each peer has to compute the number of participating peers via DHT synchronization instead of using the information received from the dispatcher
                    num_participating_peers, _ = count_num_participating_peers(peer_id, iteration, num_peers, dht)

                    # Communication and federated averaging
                    communicated_bytes = 0
                    kl_factor = 0
                    round = 1
                    group_lengths = []
                    result_1_model, result_1_momentum, integrated_count, communicated_bytes = communicate_models(TESTING, device, peer_id, num_peers, num_participating_peers, model, momentum_vector, 
                                                                                                             shared_model_dict, dht, iteration, round, communicated_bytes, validation_loader=validation_loader, validator=validator, ledger=ledger, enable_validation=enable_validation)
                    group_lengths.append(integrated_count)
                    if result_1_model:
                        # Aggregate collected models and momentum vectors
                        if isinstance(model, ModernBERTClassifier):
                            model.load_state_dict(result_1_model, strict=False)
                            trainable_params_indices = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
                            for i, global_idx in enumerate(trainable_params_indices):
                                momentum_vector[global_idx] = result_1_momentum[i]
                        else:
                            model.load_state_dict(result_1_model, strict=True)
                            momentum_vector = result_1_momentum
                    else:
                        print(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any models in round 1.")
                    del result_1_model, result_1_momentum
                    torch.cuda.empty_cache()
                    gc.collect()

                    avg_group_length = sum(group_lengths) / len(group_lengths) if group_lengths else 0
                    aggregation_duration = time.time() - start_time_aggregation
                    
                    if ledger is not None and iteration % 10 == 0:
                        stats = ledger.get_statistics()
                        print(f"[Peer {peer_id}] Ledger Stats: {stats}", flush=True)
                    
                    # If testing iteration then evaluate the model in a subprocess
                    avg_group_length = sum(group_lengths) / len(group_lengths) if group_lengths else 0
                    aggregation_duration = time.time() - start_time_aggregation

                    if do_testing:
                        start_time_testing = time.time()    
                        test_acc, test_loss = evaluate(model, device, test_loader, label="Peer Test", peer_id=peer_id)
                        testing_duration = time.time() - start_time_testing
                        result_queue.put((4, iteration, peer_id, test_acc, test_loss, (avg_group_length, communicated_bytes, kl_factor, learning_rate, aggregation_duration, testing_duration)))
                        if ml_task == "news" and isinstance(test_acc, (int, float)) and test_acc > 50:
                            learning_rate = 0.02
                    else:
                        result_queue.put((4, iteration, peer_id, 0, 0, (avg_group_length, communicated_bytes, kl_factor, learning_rate, aggregation_duration, 0)))

                    
                # Avoid peers to crash during rare participation windows
                elif task == "skip":
                    print(f"Peer {peer_id} skipping iteration {iteration}...")
                    continue
                
                # Shutdown the peer
                elif task == "shutdown":
                    print(f"Peer {peer_id} shutting down...")
                    break
            
            except (queue.Empty, ValueError):
                # Keep waiting for tasks if none are available and skip value errors
                continue
    
    # Clean shutdown
    except Exception as e:
        print(f"Peer process exception: {e}", flush=True)
    finally:
        print(f"Peer process {peer_id} exiting, cleaning up.", flush=True)
        try:
            dht.shutdown()
        except Exception:
            pass
        try:
            shared_model_dict.clear()
            del shared_model_dict
        except Exception:
            pass
        try:
            task_queue.close()
            result_queue.close()
            task_queue.join_thread()
            result_queue.join_thread()
        except Exception:
            pass
        try:
            del task_queue
            del result_queue
        except Exception:
            pass
        gc.collect()

# Periodic cleanup of the shared dictionary to avoid filling up the shared memory with stale data
def periodic_dict_cleanup(shared_dict, current_iteration):
    if current_iteration == 0:
        return
    try:
        if not shared_dict.keys():
            print(f"[Dispatcher] Periodic cleanup on iteration {current_iteration}: Shared dictionary is already empty.")
            return
        print(f"[Dispatcher Cleanup] Clearing all stale entries from previous iterations...")        
        shared_dict.clear()
        print(f"[Dispatcher Cleanup] Shared dictionary cleared successfully.")
    except Exception as e:
        print(f"[Dispatcher Cleanup] Error during shared dictionary cleanup: {e}")

# Average models and momentum vectors according to FedAvg
def fedavg_aggregation(model, models_collected, momentum_vector, momentum_vectors_collected):
    
    # 1v2: aggregate model weights
    averaged_state_dict = copy.deepcopy(models_collected[0])
    for key in averaged_state_dict:
        stacked = torch.stack([peer_model[key].float() for peer_model in models_collected])
        averaged_state_dict[key] = stacked.mean(dim=0)
    if isinstance(model, ModernBERTClassifier):
        model.load_state_dict(averaged_state_dict, strict=False)
    else:
        model.load_state_dict(averaged_state_dict, strict=True)

    # 2v2: aggregate momentum vectors
    if isinstance(model, ModernBERTClassifier):
        trainable_params_indices = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
        for i, global_idx in enumerate(trainable_params_indices):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[global_idx] = stacked.mean(dim=0)
    else:
        for i in range(len(momentum_vector)):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[i] = stacked.mean(dim=0)
    return model, momentum_vector

def dispatcher(max_runtime, ml_task, peer_processes, task_queue, result_queue, device, num_peers, 
               local_update_participation_prob, aggregation_participation_prob, peer_dropout_likelihood, max_iterations, 
               mini_batches_per_iteration, testing_frequency, convergence_threshold, convergence_patience, dispatcher_patience):
    """ Central controller dispatching tasks dynamically """
    
    print("Dispatcher assigning tasks...")
    dispatcher_start_time = time.time()
    max_runtime_seconds = max_runtime * 3600

    # Parameter initialization
    best_loss, avg_loss, avg_acc = float("inf"), float("inf"), 0
    best_iteration = 0
    historical_losses = {}
    historical_accuracies = {}
    total_iterations = max_iterations
    k_update, k_aggregate = num_peers, num_peers
    do_testing = True

    # Initialize model and momentum vector
    if device == "cuda":
        assert torch.cuda.is_available(), f"[{datetime.now()}] WARNING: Dispatcher requested CUDA but not available."
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if ml_task == "news":
        model = ModernBERTClassifier().to(device)
    elif ml_task == "mnist":
        model = SimpleCNN().to(device)
    momentum_vector = [torch.zeros_like(p, device=device) for p in model.parameters()]

    # Each iteration contains of one local model update per peer and one full mesh model aggregation
    for iteration in range(max_iterations):

        # Logging
        print(f"DISPATCHER STARTING ITERATION {iteration+1}/{max_iterations}...")
        print(datetime.now())
        start_time = time.time()

        # Track communication traffic
        iteration_sdict_bytes = 0 # NOTE truely relevant data traffic (model parameters & momentum vectors)
        iteration_tqueue_bytes, iteration_rqueue_bytes = 0, 0 # NOTE simulation-specific (dispatcher) traffic and would not occur in "productive mode" (self-controlling peers) => irrelevant for comparing with p2p baselines



        # -------------------- Client Update --------------------
        
        # Participation rates which determine heterogeneous peer participation
        if iteration >= 0: # NOTE optionally let all peers participate in the first iterations to enable initial intense global synchronization
            k_update, k_aggregate = int(num_peers * local_update_participation_prob / 100), int(num_peers * aggregation_participation_prob / 100)

        # Simulate heterogeneous peer participation
        participating_update_peers = random.sample(range(num_peers), k=k_update)
        
        # Trigger update tasks
        print("Dispatcher puts \"update\"...")
        print(len(participating_update_peers))
        print(sorted(participating_update_peers))
        queue_object_a = ("update", iteration, mini_batches_per_iteration)
        queue_object_b = ("skip", iteration, None)
        queue_object_size_a = len(pickle.dumps(queue_object_a))
        queue_object_size_b = len(pickle.dumps(queue_object_b))
        for peer in range(num_peers): # NOTE assign "update" task only to selected peers s.t. this loop functions as DSpodFL's "Sporadic SGD" term
            if peer in participating_update_peers:
                task_queue.put((peer, queue_object_a))
                iteration_tqueue_bytes += queue_object_size_a
            else:
                task_queue.put((peer, queue_object_b))
                iteration_tqueue_bytes += queue_object_size_b
        
        # Wait for completion of all participating local updates
        results = []
        training_losses = []
        training_durations = []
        max_get_difference = 120
        last_get = time.time() + dispatcher_patience
        if iteration == 0:
            max_get_difference = 10 * 120
            last_get = time.time() + 10 * dispatcher_patience
            print("Higher dispatcher patience during local updates because first FL iteration!")
        while len(results) < len(participating_update_peers) and (time.time() - last_get) < max_get_difference:
            if shutdown_event.is_set(): # NOTE added cleanup best practice
                print("Dispatcher received shutdown signal, stopping...")
                break
            try:
                result_object = result_queue.get(timeout=0.1)
                iteration_rqueue_bytes += len(pickle.dumps(result_object))
                task_id, iteration_id, peer_id, training_loss, training_message, (remaining_batches, training_duration) = result_object
                if iteration_id != iteration or task_id != 3 or isinstance(training_loss,(int,float)) == False:
                    print(f"WARNING: Skipped result queue entry while waiting for update reports: Peer {peer_id}.")
                    continue
                last_get = time.time()
                results.append(peer_id)
                training_losses.append(training_loss)
                training_durations.append(training_duration)
                print(f"{training_message} => {remaining_batches} remaining mini-batches.")
            except queue.Empty:
                continue
        missing_updates = set(participating_update_peers) - set(results)
       
        # Shutdown if missing results to avoid crashing script and leftover shared memory resources
        if missing_updates:
            print(f"WARNING: Timeout! Missing update results from peers: {missing_updates}.")
            print("DISPATCHER: Initiating shutdown due to missing peer updates.")
            for peer in range(num_peers):
                task_queue.put(("All", ("shutdown", max_iterations, 0)))
            return

        # Logging of peer status
        dead_peers = [peer_id for peer_id, p in enumerate(peer_processes) if not p.is_alive()]
        if dead_peers:
            print(f"⚠️ Dead/non-responsive peers: {dead_peers}.")
        else:
            print("🎉 All peer processes are alive.")



        # -------------------- Model Aggregation --------------------

        # Check if peers have to do model testing
        if iteration % testing_frequency == 0:
            do_testing = True
        else:
            do_testing = False

        # Simulate partial participation (aggregation_participation_prob) and network churn (peer_dropout_likelihood)
        participating_aggregation_peers = random.sample(range(num_peers), k=k_aggregate) # NOTE simulate partial participation
        if aggregation_participation_prob == local_update_participation_prob: # NOTE true partial participation (peer either participates in entire FL iteration or does not)
            participating_aggregation_peers = participating_update_peers
        if peer_dropout_likelihood > 0: # NOTE simulate network churn
            active_peers = []
            dropout_prob_float = peer_dropout_likelihood / 100.0
            for peer in participating_aggregation_peers:
                if random.random() > dropout_prob_float: # NOTE sampling from binomial distribution
                    active_peers.append(peer)
            dropped_count = len(participating_aggregation_peers) - len(active_peers)
            if dropped_count > 0:
                print(f"{dropped_count} peers dropped out of aggregation due to network churn simulation.")            
            participating_aggregation_peers = active_peers
        num_participating_aggregation_peers = len(participating_aggregation_peers)
        
        # Trigger aggregation tasks
        print("Dispatcher puts \"aggregate1\"...")
        print(num_participating_aggregation_peers)
        print(sorted(participating_aggregation_peers))
        queue_object_a = ("aggregate1", iteration, None)
        queue_object_b = ("skip", iteration, None)
        queue_object_size_a = len(pickle.dumps(queue_object_a))
        queue_object_size_b = len(pickle.dumps(queue_object_b))
        for peer in range(num_peers): # NOTE assign "aggregate1" task only to selected peers s.t. this loop functions as DSpodFL's "Sporadic aggregation" term
            if peer in participating_aggregation_peers:
                task_queue.put((peer, queue_object_a))
                iteration_tqueue_bytes += queue_object_size_a
            else:
                task_queue.put((peer, queue_object_b))
                iteration_tqueue_bytes += queue_object_size_b
        
        # Wait for all participating peer models and momentum vectors
        start_time_aggregation = time.time()
        results = []
        models_collected = []
        momentum_vectors_collected = []
        max_get_difference = 120
        last_get = time.time() + dispatcher_patience
        if iteration == 0:
            max_get_difference = 10 * 120
            last_get = time.time() + 10 * dispatcher_patience
            print("Higher dispatcher patience during aggregation because first FL iteration!")
        while len(results) < len(participating_aggregation_peers) and (time.time() - last_get) < max_get_difference:
            if shutdown_event.is_set(): # NOTE added cleanup best practice
                print("Dispatcher received shutdown signal, stopping...")
                break
            try:
                result_object = result_queue.get(timeout=0.1)
                iteration_rqueue_bytes += len(pickle.dumps(result_object))
                task_id, iteration_id, peer_id, peer_model_state_dict, peer_momentum_vector, _ = result_object
                if iteration_id != iteration or task_id != 4:
                    print(f"WARNING: Skipped result queue entry while waiting for model and momentum vector: Peer {peer_id}.")
                    continue
                last_get = time.time()
                models_collected.append(peer_model_state_dict)
                momentum_vectors_collected.append(peer_momentum_vector)
                results.append(peer_id)
            except queue.Empty:
                continue
        missing_aggregations = set(participating_aggregation_peers) - set(results)
        
        # Shutdown if missing results to avoid crashing script and leftover shared memory resources
        if missing_aggregations:
            print(f"WARNING: Timeout! Missing aggregation results 1of2 from peers: {missing_aggregations}.")
            print("DISPATCHER: Initiating shutdown due to missing peer aggregations.")
            for peer in range(num_peers):
                task_queue.put(("All", ("shutdown", max_iterations, 0)))
            return
        else:
            print(f"Received local models and momentum vectors from peers {sorted(results)}.")

        # Track communication traffic and conduct federated averaging
        iteration_sdict_bytes = sum(
            sum(p.numel() * p.element_size() for p in peer_model.values())
            for peer_model in models_collected
        )
        iteration_sdict_bytes += sum(
            sum(m.numel() * m.element_size() for m in peer_momentum)
            for peer_momentum in momentum_vectors_collected
        )
        model, momentum_vector = fedavg_aggregation(model, models_collected, momentum_vector, momentum_vectors_collected)

        # Send global model and momentum vector solely to participating aggregation peers
        print("Dispatcher puts \"aggregate2\"...")
        print(num_participating_aggregation_peers)
        print(sorted(participating_aggregation_peers))
        if isinstance(model, ModernBERTClassifier):
            cpu_model = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            cpu_momentum_vector = [t.cpu().clone() for p, t in zip(model.parameters(), momentum_vector) if p.requires_grad]
        else:
            cpu_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            cpu_momentum_vector = [t.cpu().clone() for t in momentum_vector]
        queue_object_values = (cpu_model, cpu_momentum_vector, do_testing)
        queue_object = ("aggregate2", iteration, queue_object_values)
        queue_object_size = len(pickle.dumps(queue_object))
        queue_object_b = ("skip", iteration, None)
        queue_object_size_b = len(pickle.dumps(queue_object_b))
        for peer in range(num_peers): # NOTE send global model and momentum vector solely to participating aggregation peers
            if peer in participating_aggregation_peers:
                task_queue.put((peer, queue_object))
                iteration_tqueue_bytes += queue_object_size
            else:
                task_queue.put((peer, queue_object_b))
                iteration_tqueue_bytes += queue_object_size_b
        aggregation_duration = time.time() - start_time_aggregation

        # Track communication traffic
        global_model_bytes = sum(t.numel() * t.element_size() for t in cpu_model.values())
        global_mom_bytes   = sum(t.numel() * t.element_size() for t in cpu_momentum_vector)
        iteration_sdict_bytes += (global_model_bytes + global_mom_bytes) * num_participating_aggregation_peers
        
        # Wait for completion of all model evaluations
        results = []
        losses = []
        accs = []
        learning_rates = []
        testing_durations = []
        kl_factor = 0
        max_get_difference = 120
        last_get = time.time() + dispatcher_patience
        if iteration == 0:
            max_get_difference = 10 * 120
            last_get = time.time() + 10 * dispatcher_patience
            print("Higher dispatcher patience during aggregation because first FL iteration!")
        while len(results) < len(participating_aggregation_peers) and (time.time() - last_get) < max_get_difference:
            if shutdown_event.is_set(): # NOTE added cleanup best practice
                print("Dispatcher received shutdown signal, stopping...")
                break
            try:
                result_object = result_queue.get(timeout=0.1)
                iteration_rqueue_bytes += len(pickle.dumps(result_object))
                task_id, iteration_id, peer_id, acc, loss, (peer_lr, testing_duration) = result_object
                if iteration_id != iteration or task_id != 5 or isinstance(acc,(int,float)) == False or isinstance(loss,(int,float)) == False:
                    print(f"WARNING: Skipped result queue entry while waiting for aggregation reports: Peer {peer_id}.")
                    continue
                if do_testing:
                    acc_f = float(acc)
                    loss_f = float(loss)
                    accs.append(acc_f)
                    losses.append(loss_f)
                    print(f"Peer {peer_id} - Test accuracy: {acc_f:.2f}% & Loss: {loss_f:.4f}.")
                last_get = time.time()
                results.append(peer_id)
                learning_rates.append(peer_lr)
                testing_durations.append(testing_duration)
            except queue.Empty:
                continue
        missing_aggregations = set(participating_aggregation_peers) - set(results)
        
        # Shutdown if missing results to avoid crashing script and leftover shared memory resources
        if missing_aggregations:
            print(f"WARNING: Timeout! Missing aggregation results 2of2 from peers: {missing_aggregations}.")
            print("DISPATCHER: Initiating shutdown due to missing peer aggregations.")
            for peer in range(num_peers):
                task_queue.put(("All", ("shutdown", max_iterations, 0)))
            return
        else:
            print(f"Received indication for completion of model evaluation from peers {sorted(results)}.")

        # Logging
        avg_training_loss = sum(training_losses) / len(training_losses) if training_losses else 10
        avg_training_duration = sum(training_durations) / len(training_durations) if training_durations else 0
        avg_lr = sum(learning_rates) / len(learning_rates) if learning_rates else 0
        avg_testing_duration = sum(testing_durations) / len(testing_durations) if testing_durations else 0
        if do_testing:
            avg_loss = sum(losses) / len(losses) if losses else 10
            avg_acc = sum(accs) / len(accs) if accs else 0
            historical_losses[iteration] = avg_loss
            historical_accuracies[iteration] = avg_acc
        total_time = time.time() - start_time
        print(f"Iteration {iteration+1} finished at {datetime.now()}.")
        print(f"Iteration duration: {total_time:.2f} seconds.")
        wandb.log({
            "iteration": iteration+1,
            "participating_update_peers": len(participating_update_peers),
            "participating_aggregation_peers": len(participating_aggregation_peers),
            "dead_peers": len(dead_peers),
            "avg_training_loss": avg_training_loss,
            "testing_loss": avg_loss,
            "testing_accuracy": avg_acc,
            "iteration_tqueue_bytes": iteration_tqueue_bytes,
            "iteration_rqueue_bytes": iteration_rqueue_bytes,
            "iteration_sdict_bytes": iteration_sdict_bytes,
            "iteration_duration": total_time,
            "kl_factor": kl_factor,
            "avg_learning_rate": avg_lr,
            "avg_training_duration": avg_training_duration,
            "aggregation_duration": aggregation_duration,
            "avg_testing_duration": avg_testing_duration
        })

        # Early stopping check
        if do_testing:

            # Stop if target accuracy is reached
            stop_acc = None
            if ml_task.lower() == "mnist":
                stop_acc = 95.0
            elif ml_task.lower() == "news":
                stop_acc = 50.0
            if stop_acc is not None and avg_acc >= stop_acc:
                total_iterations = iteration + 1
                print(f"Early stopping triggered at round {total_iterations}: Reached {avg_acc:.2f}% accuracy (threshold: {stop_acc}%)")
                break

            # Stop if loss has not improved compared to 100 iterations ago
            check_iteration = iteration - 100
            if check_iteration in historical_losses:
                previous_loss = historical_losses[check_iteration]
                if avg_loss >= previous_loss:
                    total_iterations = iteration + 1
                    print(f"Early stopping triggered at iteration {total_iterations}: Current loss ({avg_loss:.4f}) is not better than loss at iteration {check_iteration + 1} ({previous_loss:.4f}).")
                    break

            # Stop if accuracy has not improved compared to 100 iterations ago
            if check_iteration in historical_accuracies:
                previous_acc = historical_accuracies[check_iteration]
                if avg_acc <= previous_acc:
                    total_iterations = iteration + 1
                    print(f"Early stopping triggered at iteration {total_iterations}: Current accuracy ({avg_acc:.2f}%) is not better than accuracy at iteration {check_iteration + 1} ({previous_acc:.2f}%).")
                    break

            # Stop if the job's max runtime is approaching
            elapsed_time = time.time() - dispatcher_start_time
            if elapsed_time >= (max_runtime_seconds - 20 * 60):
                total_iterations = iteration + 1
                print(f"Early stopping triggered at iteration {total_iterations}: Time limit approaching. Elapsed time: {elapsed_time:.2f}s, Max runtime: {max_runtime_seconds:.2f}s.")
                break

    # Logging
    wandb.log({
        "total_iterations": total_iterations,
        "best_testing_loss": best_loss,
        "best_iteration": best_iteration
    })

    # Add shutdown tasks
    for peer in range(num_peers):
        task_queue.put(("All", ("shutdown", max_iterations, 0)))

def main():
    """ Spawns NUM_PEERS processes running as persistent peers """

    # Shared memory check before any multiprocessing
    print("[SHM-CHECK][BEFORE MP] /dev/shm usage before starting Manager/processes:")
    shm_before_mp_MB = _dev_shm_usage_bytes() / 1024 / 1024
    shm_before_mp_total_MB = _dev_shm_total_usage_bytes() / 1024 / 1024
    print(f"    /dev/shm (psm_, torch_): ≈ {shm_before_mp_MB:.1f} MB")
    print(f"    /dev/shm (ALL): ≈ {shm_before_mp_total_MB:.1f} MB\n")
    wandb.log({
        "shm_before_mp_MB": shm_before_mp_MB,
        "shm_before_mp_total_MB": shm_before_mp_total_MB,
    })

    # Context manager for multiprocessing.Manager()
    ctx = tmp.get_context('spawn')
    with ctx.Manager() as manager:
        shared_model_dict = manager.dict()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Start a single bootstrap node
        bootstrap_dht = hivemind.DHT(initial_peers=[], start=True)
        bootstrap_address = bootstrap_dht.get_visible_maddrs()
        start_time = time.time()
        print(f"Bootstrap node started on {bootstrap_address} at {datetime.now()}.")

        # Get list of available CPU core indices
        valid_cores = list(os.sched_getaffinity(0))

        # Preload the model to avoid Huggingface access issues in case of using BERT
        ml_task = wandb.config.ml_task
        print(f"Main process is pre-loading the '{ml_task}' model...")
        if ml_task == "news":
            model = ModernBERTClassifier()
        elif ml_task == "mnist":
            model = SimpleCNN()
        print("Model instance is created.")

        # Create persistent peer processes
        peer_processes = [

            ctx.Process(target=peer_process, args=(
                wandb.config.seed,
                wandb.config.split_type,
                wandb.config.dirichlet_alpha,
                ml_task,
                i,
                bootstrap_address, 
                task_queue, 
                result_queue, 
                wandb.config.num_peers, 
                wandb.config.peers_per_core,
                wandb.config.learning_rate,
                wandb.config.momentum, 
                shared_model_dict,
                valid_cores,
                wandb.config.device,
                copy.deepcopy(model),
                None,
                None,
                wandb.config.enable_validation,
                wandb.config.accuracy_threshold,
                wandb.config.loss_threshold,
            ))

            for i in range(wandb.config.num_peers)
        ]

        # Start all peer processes
        for p in peer_processes:
            p.start()

        try:
            # Run dispatcher to assign tasks
            dispatcher(
                max_runtime=wandb.config.max_runtime,
                ml_task=ml_task,
                shared_model_dict=shared_model_dict,
                peer_processes=peer_processes, 
                task_queue=task_queue, 
                result_queue=result_queue, 
                bootstrap_dht=bootstrap_dht,
                valid_cores=valid_cores,
                device=wandb.config.device,
                num_peers=wandb.config.num_peers, 
                peers_per_core=wandb.config.peers_per_core, 
                local_update_participation_prob=wandb.config.local_update_participation_prob,
                aggregation_participation_prob=wandb.config.aggregation_participation_prob,
                peer_dropout_likelihood=wandb.config.peer_dropout_likelihood,
                max_iterations=wandb.config.max_iterations,
                mini_batches_per_iteration=wandb.config.mini_batches_per_iteration, 
                testing_frequency=wandb.config.testing_frequency,
                convergence_threshold=wandb.config.convergence_threshold,
                convergence_patience=wandb.config.convergence_patience,
                dispatcher_patience=wandb.config.dispatcher_patience,
                learning_rate=wandb.config.learning_rate,
                momentum=wandb.config.momentum,
            )
        except Exception as e:
            print(f"Main exception: {e}", flush=True)
        finally:
            print("Main cleanup: Notifying all peers to shut down...", flush=True) # NOTE added cleanup best practice
            
            # 1v4: notify: sk all peers to shut down gracefully.
            for i in range(wandb.config.num_peers):
                try:
                    task_queue.put(("All", ("shutdown", -1, 0)), timeout=0.1)
                except queue.Full:
                    print("Could not send shutdown task to all peers, queue is full.")
                    break
            print("Main cleanup: Waiting for peer processes to join...", flush=True)
            
            # 2v4: join: wait for processes to exit on their own.
            for p in peer_processes:
                p.join(timeout=10)
            print("Main cleanup: Terminating any remaining processes...", flush=True)

            # 3v4: terminate: forcefully stop any process that is still running.
            for p in peer_processes:
                if p.is_alive():
                    print(f"Peer process {p.pid} did not exit gracefully. Terminating.")
                    p.terminate()
                    p.join(timeout=60)

            # 4v4: shutdown other resources
            try:
                print("Main cleanup: Clearing all entries from the shared dictionary...")
                shared_model_dict.clear()
                print("Main cleanup: Shared dictionary cleared.")
            except Exception as e:
                print(f"Main cleanup: Could not clear shared_model_dict. Reason: {e}")
            try:
                task_queue.close()
                result_queue.close()
                task_queue.join_thread()
                result_queue.join_thread()
            except Exception:
                pass
            try:
                manager.shutdown()
            except Exception:
                pass
            try:
                del shared_model_dict
                del task_queue
                del result_queue
            except Exception:
                pass
            try:
                bootstrap_dht.shutdown()
            except Exception:
                pass
            gc.collect()
            total_time = time.time() - start_time
            print(f"All peers have completed their tasks at {datetime.now()}.")
            print(f"Total duration: {total_time:.2f} seconds.")

        # Kill all zombie processes
        if ADMIN:
            [os.kill(int(ppid), 9) for ppid in set(subprocess.getoutput("ps -eo ppid,state | awk '$2==\"Z\" {print $1}'").split()) if ppid.isdigit() and int(ppid) > 1]

        # Logging
        total_time = time.time() - start_time
        print(f"All peers have completed their tasks at {datetime.now()}.")
        print(f"Total duration: {total_time:.2f} seconds.")

    # Shared memory check after multiprocessing cleanup
    print("[SHM-CHECK][AFTER MP] /dev/shm usage after closing all multiprocessing resources:")
    shm_after_mp_MB = _dev_shm_usage_bytes() / 1024 / 1024
    shm_after_mp_total_MB = _dev_shm_total_usage_bytes() / 1024 / 1024
    print(f"    /dev/shm (psm_, torch_): ≈ {shm_after_mp_MB:.1f} MB")
    print(f"    /dev/shm (ALL): ≈ {shm_after_mp_total_MB:.1f} MB\n")
    wandb.log({
        "shm_after_mp_MB": shm_after_mp_MB,
        "shm_after_mp_total_MB": shm_after_mp_total_MB,
    })

    # Compute and log the left over and added shared memory allocation caused by my job
    shm_diff_MB = shm_after_mp_MB - shm_before_mp_MB
    shm_diff_total_MB = shm_after_mp_total_MB - shm_before_mp_total_MB
    print(f"[SHM-CHECK][DIFF] Difference in /dev/shm allocation (psm_, torch_): {shm_diff_MB:.1f} MB")
    print(f"[SHM-CHECK][DIFF] Difference in /dev/shm allocation (ALL): {shm_diff_total_MB:.1f} MB")
    wandb.log({
        "shm_diff_after_before_MB": shm_diff_MB,
        "shm_diff_after_before_total_MB": shm_diff_total_MB,
    })

if __name__ == "__main__":

    # Kill all zombie processes
    if ADMIN:
        [os.kill(int(ppid), 9) for ppid in set(subprocess.getoutput("ps -eo ppid,state | awk '$2==\"Z\" {print $1}'").split()) if ppid.isdigit() and int(ppid) > 1]

    # Argument parsing
    parser = argparse.ArgumentParser(description="AR Peer-to-Peer Federated Learning")
    parser.add_argument("--sd", type=int, default=42, help="Seed for random value generation")
    parser.add_argument("--ml", type=str, default="mnist", help="ML task (mnist or news)")
    parser.add_argument("--dv", type=str, default="cuda", help="Device to use (cpu or cuda)")
    parser.add_argument("--np", type=int, default=64, help="Number of peers")
    parser.add_argument("--pp", type=int, default=1, help="Peers per core")
    parser.add_argument("--up", type=int, default=100, help="Local update participation probability (between 0 and 100)")
    parser.add_argument("--ap", type=int, default=100, help="Aggregation participation probability (between 0 and 100)")
    parser.add_argument("--dl", type=int, default=0, help="Peer dropout likelihood (between 0 and 100)")
    parser.add_argument("--iv", type=int, default=4, help="Iterations mechanism value (lower if higher number of peers)")
    parser.add_argument("--rt", type=int, default=12, help="Maximum allowed runtime in hours")
    parser.add_argument("--sp", type=str, default="iid", choices=["iid","dirichlet"], help="How to split training data across peers")
    parser.add_argument("--al", type=float, default=1.0, help="Dirichlet alpha for non-IID when --sp=dirichlet")
    parser.add_argument("--ev", type=int, default=1, help="Enable validation (1=enabled, 0=disabled for standard FedAvg)")
    parser.add_argument("--at", type=float, default=0.0, help="Accuracy threshold for accepting updates (percentage points)")
    parser.add_argument("--lt", type=float, default=0.0, help="Loss threshold for accepting updates")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed = args.sd
    set_all_seeds(seed)

    # Overwrite globals from arguments
    ML_TASK = args.ml
    DEVICE = args.dv
    NUM_PEERS = args.np
    PEERS_PER_CORE = args.pp
    LOCAL_UPDATE_PARTICIPATION_PROB = args.up
    AGGREGATION_PARTICIPATION_PROB = args.ap
    PEER_DROPOUT_LIKELIHOOD = args.dl
    ITERATIONS_MECHANISM_VALUE = args.iv
    max_runtime = args.rt
    ENABLE_VALIDATION = bool(args.ev)
    ACCURACY_THRESHOLD = args.at
    LOSS_THRESHOLD = args.lt

    if ML_TASK == "news":
        if args.sp == "dirichlet":
            mock_train_loader, _, _ = get_text_data_loaders_dirichlet(partition=0, num_partitions=NUM_PEERS, seed=seed, alpha=args.al)
        else:
            mock_train_loader, _, _ = get_text_data_loaders(partition=0, num_partitions=NUM_PEERS, seed=seed)
    elif ML_TASK == "mnist":
        if args.sp == "dirichlet":
            mock_train_loader, _, _ = get_mnist_data_loaders_dirichlet(partition=0, num_partitions=NUM_PEERS, seed=seed, alpha=args.al)
        else:
            mock_train_loader, _, _ = get_mnist_data_loaders(partition=0, num_partitions=NUM_PEERS, seed=seed)
    TOTAL_BATCHES = len(mock_train_loader)

    # Different project name if non-IID data
    PROJECT = "p2p_fl_niid"

    # Modify some hyperparameters when doing BERT
    if ML_TASK == "news":
        MAX_ITERATIONS = 1000
    
    # Determine mini-batches per iteration
    mini_batches_per_iteration = 0
    if ITERATIONS_MECHANISM_APPROACH == 0:
        mini_batches_per_iteration = ITERATIONS_MECHANISM_VALUE
    elif ITERATIONS_MECHANISM_APPROACH == 1:
        mini_batches_per_iteration = math.ceil(TOTAL_BATCHES / ITERATIONS_MECHANISM_VALUE)
    else:
        raise RuntimeError("ERROR: Invalid iterations mechanism approach!")
    
    # WandB config
    unique_run_name = generate_unique_run_name(args.sp, ENTITY, PROJECT, seed, ML_TASK, NUM_PEERS, mini_batches_per_iteration, AGGREGATION_PARTICIPATION_PROB, LOCAL_UPDATE_PARTICIPATION_PROB, PEER_DROPOUT_LIKELIHOOD)
    wandb_dir = "/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ga27fed2/wandb_runs"
    run = wandb.init(
        dir=wandb_dir,
        entity=ENTITY,
        project=PROJECT,
        name=unique_run_name,
        config={
            'seed': seed,
            'ml_task': ML_TASK,
            'device': DEVICE,
            'num_peers': NUM_PEERS,
            'peers_per_core': PEERS_PER_CORE,
            'local_update_participation_prob': LOCAL_UPDATE_PARTICIPATION_PROB,
            'aggregation_participation_prob': AGGREGATION_PARTICIPATION_PROB,
            'peer_dropout_likelihood': PEER_DROPOUT_LIKELIHOOD,
            'max_iterations': MAX_ITERATIONS,
            'mini_batches_per_iteration': mini_batches_per_iteration,
            'learning_rate': LEARNING_RATE,
            'momentum': MOMENTUM,
            'testing_frequency': TESTING_FREQUENCY,
            'convergence_patience': CONVERGENCE_PATIENCE,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
            'dispatcher_patience': DISPATCHER_PATIENCE,
            'max_runtime': max_runtime,
            'split_type': args.sp,
            'dirichlet_alpha': args.al,
            'enable_validation': ENABLE_VALIDATION,
            'accuracy_threshold': ACCURACY_THRESHOLD,
            'loss_threshold': LOSS_THRESHOLD,
        },
    )

    # Start processes via spawn instead of fork
    tmp.set_start_method("spawn", force=True)

    # Let's go
    main()

# ==================== VERSION INFO ====================

__version__ = '1.0.0'
__all__ = ['UtilityValidator', 'ModelLedger', 'create_validation_components']
