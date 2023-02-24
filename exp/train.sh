#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --dataset cdnyt \
        --output_dir output \
        --pre_encoder bert-base-uncased \
        --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --weight_decay 1e-2 \
        --gradient_clip_val 1 \
        --temperature 10. \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --test_batch_size 64 \
        --max_length 60 \
        --max_epochs 1 \
        --save_top_k 1 \
        --gpus 1 \
        --num_sanity_val_steps -1 \
        --log_every_n_steps 25 \
        --val_check_interval 125 \
        --num_workers 24 \
        --wandb \
        --fp16