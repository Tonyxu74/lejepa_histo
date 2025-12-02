#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=20         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G                  # more workers (cpus) require more ram
#SBATCH --time=0-10:00:00

source /home/txu/miniconda3/etc/profile.d/conda.sh

conda activate lymph_proj


python /home/txu/lejepa_histo/train_supervised.py \
  --train_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_train.json'\
  --val_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_val.json' \
  --test_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_test.json' \
  --fold_json '' \
  --output_path /project/amgrp/txu/lejepa/sup_logs_ep_0 \
  --num_folds 1 \
  --pretrained_path '' \
  --use_pretrained 0 \
  --resnet_name 'resnet50' \
  --device 0 \
  --workers 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 20 \
  --num_classes 9



for ep in 9 19 29 39 49 59 69 79 89 99 ;
do

python /home/txu/lejepa_histo/train_supervised.py \
  --train_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_train.json'\
  --val_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_val.json' \
  --test_json '/aippmdata/public/NCT-CRC-HE-100K/nct-crc-he-100k_test.json' \
  --fold_json '' \
  --output_path /project/amgrp/txu/lejepa/sup_logs_ep_$ep \
  --num_folds 1 \
  --pretrained_path /project/amgrp/txu/lejepa/logs/model_weights_epoch_${ep}.pt \
  --use_pretrained 1 \
  --resnet_name 'resnet50' \
  --device 0 \
  --workers 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 20 \
  --num_classes 9

done
