#!/usr/bin/env bash
# python dnn-ga.py --result_save_dir=1010/1 --EPOCHS=100 --N_POPULATION=100 --N_BEST=10 --h1=30 --h2=30 --h3=30
# python dnn-ga.py --result_save_dir=1010/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128

# ===score function changed ===
# python dnn-ga.py --result_save_dir=1011/1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=256 --h2=256 --h3=256 --early_stopping=50
# python dnn-ga.py --result_save_dir=1011/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --h1=50 --h2=50 --h3=50 --early_stopping=30
# python dnn-ga.py --result_save_dir=1011/3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --h1=128 --h2=128 --h3=128 --early_stopping=50
# python dnn-ga.py --result_save_dir=1011/4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --h1=128 --h2=128 --h3=128 --early_stopping=50
# python dnn-ga.py --result_save_dir=1011/5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --h1=128 --h2=128 --h3=128 --early_stopping=50

# change linear to relu
# python dnn-ga.py --result_save_dir=1011/6 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --h1=512 --h2=512 --h3=512 --early_stopping=50
# python dnn-ga.py --result_save_dir=1011/7 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=512 --h2=512 --h3=512 --early_stopping=300

# == score function option changed ===
# python dnn-ga.py --result_save_dir=1012/1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=0
# python dnn-ga.py --result_save_dir=1012/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=1
# python dnn-ga.py --result_save_dir=1012/3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=2
# python dnn-ga.py --result_save_dir=1012/4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=3
python dnn-ga.py --result_save_dir=1012/5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=1