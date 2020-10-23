#!/usr/bin/env bash

for i in $(seq 1 1 27)
do
  c_val=0.5
  for j in $(seq 17 1 21)
  do
    arg1="--result_save_dir=1021/$j"
    arg2="--reference=$i"
    python test.py "$arg1" "$arg2" --score_type=6 --POWER_RATIO=0.5 --CONST="$c_val"
    c_val=`echo $c_val + 0.1|bc`
  done
done