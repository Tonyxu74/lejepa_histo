# A simple implementation of LeJEPA for histopathology

This repo is a simple reimplementation of [LeJEPA](https://github.com/rbalestr-lab/lejepa/tree/main) specifically for histopathology image analysis. 
I used standard PyTorch along with a basic trainer I created, borrowing implementation from the original codebase's [MINIMAL.md](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md) file and pseudocode from the [original paper](https://arxiv.org/pdf/2511.08544).

The code is currently implemented for simple CNN architectures (ResNets), supports multicrops and DDP, and performs logging using wandb.  

## Key Requirements

- python ≥ 3.8
- torch ≥ 1.13.0
- torchvision ≥ 0.14.0
- wandb

## Using the code

Scripts are provided to show examples for pretraining and downstream supervised finetuning for a SLURM cluster environment. 
Generic arguments can be found in `utils/default_args.py` and task-specific arguments are in respective train scripts.

## Results (in progress!)

I pretrained LeJEPA on patches taken from the [SemiCOL](https://www.semicol.org/) challenge dataset and finetuned on [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) for tissue classification.
I used a batch size of 256, 2 global crops and 8 local crops, and trained for 100 epochs. It took about 46 GB of GPU memory and 3 days to train.

I will be using SimCLR with similar settings as a baseline for comparison, along with training LeJEPA on NCT-CRC-HE-100K directly (unlabelled), and training LeJEPA using a ViT architecture.

| Pretraining Method | Pretraining Dataset                                              | Architecture | Results (Acc.)      |
|--------------------|------------------------------------------------------------------|--------------|---------------------|
| LeJEPA             | Patches taken from [SemiCOL](https://www.semicol.org/) challenge | ResNet50     | 0.948               |
| SimCLR             | Patches taken from [SemiCOL](https://www.semicol.org/) challenge | ResNet50     |                     |
| LeJEPA             | NCT-CRC-HE-100K                                                  | ResNet50     |                     |
| LeJEPA             | Patches taken from [SemiCOL](https://www.semicol.org/) challenge | ViT-Base     |                     |
| LeJEPA             | NCT-CRC-HE-100K                                                  | ViT-Base     |                     |


## Acknowledgements

Big thanks to the excellent original work that this repo is based on! This is the [LeJEPA codebase](https://github.com/rbalestr-lab/lejepa/blob/main/) and the [original paper](https://arxiv.org/pdf/2511.08544).