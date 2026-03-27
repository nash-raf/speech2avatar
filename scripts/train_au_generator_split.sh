#!/bin/bash
python generator/train.py \
    --train_dataset_path /workspace/generator_dataset_split/train \
    --val_dataset_path /workspace/generator_dataset_split/val \
    --exp_name au_finetune_split_v1 \
    --batch_size 16 \
    --iter 500000 \
    --lr 1e-5 \
    --resume_ckpt ./checkpoints/generator.ckpt \
    --num_aus 17 \
    --au_dropout_prob 0.1 \
    --static_pose_aug_prob 1.0 \
    --freeze_first_n_blocks 4 \
    --save_freq 10000 \
    --display_freq 7500
