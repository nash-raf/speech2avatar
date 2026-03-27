#!/bin/bash
python generator/train.py \
    --dataset_path /home/user/D/generator_dataset \
    --exp_name au_finetune_v1 \
    --batch_size 16 \
    --iter 500000 \
    --lr 1e-5 \
    --resume_ckpt ./checkpoints/generator.ckpt \
    --num_aus 17 \
    --au_dropout_prob 0.1 \
    --static_pose_aug_prob 0.3 \
    --freeze_first_n_blocks 4 \
    --save_freq 10000 \
    --display_freq 5000
