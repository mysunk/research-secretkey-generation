#!/usr/bin/env bash
python main.py --result_save_dir=20201208 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --POWER_RATIO=0.5 --CONST=0.8
python evaluate.py --result_save_dir=20201208 --reference=1 --POWER_RATIO=0.5 --CONST=0.8