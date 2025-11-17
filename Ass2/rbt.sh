#!/bin/bash
#SBATCH --gres=gpu:L40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o output/big_train.out
#SBATCH -e output/big_train.err
#SBATCH -p long

cd /data/users/melkerr/NLP
python3 -m Ass2.run_big_train