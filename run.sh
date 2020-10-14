#!/usr/bin/env bash
# change dataset from (1,3,5) to 1-11
python dnn-ga.py --result_save_dir=1014/2-1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.05
python dnn-ga.py --result_save_dir=1014/2-2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.1
python dnn-ga.py --result_save_dir=1014/2-3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --N_CHILDREN=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.05
python dnn-ga.py --result_save_dir=1014/2-4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --N_CHILDREN=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.1
python dnn-ga.py --result_save_dir=1014/2-5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.05
python dnn-ga.py --result_save_dir=1014/2-6 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.1
python dnn-ga.py --result_save_dir=1014/2-7 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --N_CHILDREN=20 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.05
python dnn-ga.py --result_save_dir=1014/2-8 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=20 --N_CHILDREN=20 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4 --PROB_MUTATION=0.1


# =============================================================================================================================
# Previous simulations
# =============================================================================================================================

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
# change dataset from (1,3,5) to (1-9)
# python dnn-ga.py --result_save_dir=1012/1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=0
# python dnn-ga.py --result_save_dir=1012/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=1
# python dnn-ga.py --result_save_dir=1012/3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=2
# python dnn-ga.py --result_save_dir=1012/4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=3
# python dnn-ga.py --result_save_dir=1012/5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=1
# python dnn-ga.py --result_save_dir=1013/1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=1
# python dnn-ga.py --result_save_dir=1013/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=2
# python dnn-ga.py --result_save_dir=1013/3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=3
# python dnn-ga.py --result_save_dir=1013/4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=4
# python dnn-ga.py --result_save_dir=1013/5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=5
# python dnn-ga.py --result_save_dir=1013/6 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=6
# python dnn-ga.py --result_save_dir=1013/7 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=7
# python dnn-ga.py --result_save_dir=1013/8 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=8
# python dnn-ga.py --result_save_dir=1013/9 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=9
# python dnn-ga.py --result_save_dir=1013/10 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=10
# python dnn-ga.py --result_save_dir=1013/11 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=11
# python dnn-ga.py --result_save_dir=1013/12 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=50 --score_type=12
# python dnn-ga.py --result_save_dir=1013/13 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=50 --score_type=13
# python dnn-ga.py --result_save_dir=1013/14 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=128 --h2=128 --h3=128 --early_stopping=100 --score_type=10
# python dnn-ga.py --result_save_dir=1013/15 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=256 --h2=256 --h3=256 --early_stopping=50 --score_type=13
# python dnn-ga.py --result_save_dir=1013/16 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=256 --h2=256 --h3=256 --early_stopping=100 --score_type=10
# python dnn-ga.py --result_save_dir=1013/17 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=512 --h2=512 --h3=512 --early_stopping=50 --score_type=13
# python dnn-ga.py --result_save_dir=1013/18 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=512 --h2=512 --h3=512 --early_stopping=100 --score_type=10

# 10-13-2
# ===  Change score function option from 1 ===
# python dnn-ga.py --result_save_dir=1013_2/1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=1
# python dnn-ga.py --result_save_dir=1013_2/2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=2
# python dnn-ga.py --result_save_dir=1013_2/3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=3
# python dnn-ga.py --result_save_dir=1013_2/4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4
# python dnn-ga.py --result_save_dir=1013_2/5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=5
# python dnn-ga.py --result_save_dir=1013_2/6 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=6
# python dnn-ga.py --result_save_dir=1013_2/7 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=7
# python dnn-ga.py --result_save_dir=1013_2/8 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=8
# python dnn-ga.py --result_save_dir=1013_2/4_2 --EPOCHS=1000 --N_POPULATION=50 --N_BEST=5 --N_CHILDREN=5 --h1=128 --h2=128 --h3=128 --early_stopping=100 --score_type=4

# 10-14
# change dataset from (1-9) to (1,3,5)
# python dnn-ga.py --result_save_dir=1014/1-1 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4
# python dnn-ga.py --result_save_dir=1014/1-2 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=64 --h2=64 --h3=64 --early_stopping=100 --score_type=4
# python dnn-ga.py --result_save_dir=1014/1-3 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=16 --h2=16 --h3=16 --early_stopping=100 --score_type=4 --random_seed=0
# python dnn-ga.py --result_save_dir=1014/1-4 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=0
#python dnn-ga.py --result_save_dir=1014/1-5 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=0 --PROB_MUTATION=0.05
#python dnn-ga.py --result_save_dir=1014/1-6 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=10 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=1
#python dnn-ga.py --result_save_dir=1014/1-7 --EPOCHS=1000 --N_POPULATION=50 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=2
#python dnn-ga.py --result_save_dir=1014/1-8 --EPOCHS=1000 --N_POPULATION=150 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=3
#python dnn-ga.py --result_save_dir=1014/1-9 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=0 --mutation_std=2
#python dnn-ga.py --result_save_dir=1014/1-10 --EPOCHS=1000 --N_POPULATION=100 --N_BEST=10 --N_CHILDREN=20 --h1=32 --h2=32 --h3=32 --early_stopping=100 --score_type=4 --random_seed=0 --PROB_MUTATION=0.1
