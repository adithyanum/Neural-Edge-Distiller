#!/bin/bash

mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data data/training \
  --iters 100 \
  --batch-size 2 \
  --num-layers 8 \
  --learning-rate 1e-4 \
  --adapter-path models/adapters/