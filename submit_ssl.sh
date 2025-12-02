#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=20         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G                  # more workers (cpus) require more ram
#SBATCH --time=0-80:00:00

source /home/txu/miniconda3/etc/profile.d/conda.sh

conda activate lymph_proj

#srun python -m torch.distributed.launch --nproc_per_node 1 --master_port 29502 /home/txu/lejepa_histo/train_ssl.py \
python /home/txu/lejepa_histo/train_ssl.py \
  --train_json '/project/amgrp/txu/datalists/patch_datalists/SemiCOL/unsupervised_data_fixed.json'\
  --val_json '' \
  --test_json '' \
  --fold_json '' \
  --output_path '/project/amgrp/txu/lejepa/simclr_logs' \
  --num_folds 1 \
  --pretrained_path '' \
  --use_pretrained 0 \
  --resnet_name 'resnet50' \
  --device 0 \
  --workers 8 \
  --batch_size 256 \
  --support_batch_size 16 \
  --ssl_method 'simclr' \
  --temperature 0.1 \
  --target_sharpen 0.25 \
  --support_json '' \
  --multicrop 0 \
  --warmup 10 \
  --start_lr 0.3 \
  --final_lr 0.064 \
  --lr 6.4 \
  --num_epochs 100

