#!/bin/bash

mlx_lm.fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path models/adapters/ \
  --save-path models/neural-edge-3b/