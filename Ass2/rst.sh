#!/bin/bash
#SBATCH --gres=gpu:L4:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o Ass2/small_train.out
#SBATCH -e Ass2/small_train.err
#SBATCH -p short

python3 run_small_train.py