#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o eresults.out
#SBATCH -e eresults.err

cd /data/users/melkerr/NLP
python3 -m Ass2.evala2