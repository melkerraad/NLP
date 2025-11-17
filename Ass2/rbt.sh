#!/bin/bash
#SBATCH --gres=gpu:L4:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o big_output/big_train.out
#SBATCH -e big_output/big_train.err
#SBATCH -p short

cd /data/users/melkerr/NLP
python3 -m Ass2.run_big_train