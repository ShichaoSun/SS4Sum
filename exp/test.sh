#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/test.py \
        --dataset cd \
        --output_dir best \
        --pre_encoder bert-base-uncased \
        --ckpt best/rouge2=13.7600.ckpt \
        --ext_sents 3 \
        --eval_batch_size 64 \
        --test_batch_size 64 \
        --max_length 60 \
        --gpus 1 \
        --num_sanity_val_steps 0 \
        --num_workers 24 \
        --fp16