# P2PFLPersonalized
Personalized P2P FL

## Dependency Guide

Use Python3.8.

Souce code modified for our use case using the original source code for a removed repo
Install Python packages in the following order:
1. `pip install hivemind`
2. `pip install psutil`
3. `pip uninstall torch hivemind protobuf -y`
4. `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`
5. `pip install hivemind==1.1.9`
6. `pip install protobuf==3.20.1`
7. `pip install wandb`
8. `pip install matplotlib`
9. `pip install transformers datasets`