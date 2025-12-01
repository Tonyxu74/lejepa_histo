#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=28         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G                  # more workers (cpus) require more ram
#SBATCH --time=0-72:00:00

source /home/txu/miniconda3/etc/profile.d/conda.sh

conda activate lymph_proj

#srun python -m torch.distributed.launch --nproc_per_node 1 --master_port 29502 /home/txu/lejepa_histo/train_lejepa.py \
python /home/txu/lejepa_histo/train_lejepa.py \
  --train_json '/project/amgrp/txu/datalists/patch_datalists/SemiCOL/unsupervised_data_fixed.json'\
  --val_json '' \
  --test_json '' \
  --fold_json '' \
  --output_path '/project/amgrp/txu/lejepa/logs' \
  --num_folds 1 \
  --pretrained_path '' \
  --use_pretrained 0 \
  --resnet_name 'resnet50' \
  --embedding_size 512 \
  --device 0 \
  --workers 16 \
  --batch_size 256 \
  --multicrop 8 \
  --warmup 10 \
  --lr 2e-3 \
  --lambd 0.05 \
  --num_epochs 100 \
  --save_interval 10

