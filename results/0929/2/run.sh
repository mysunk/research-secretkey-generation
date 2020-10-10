#!/usr/bin/env bash
# modify load dataset
python main.py --batch_size 128 --epochs 100 --learning_rate 1e-1 --save_dir results/0929/2 --lr_decay_rate 0.5 --lr_step_size 10