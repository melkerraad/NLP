#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o outputs/tokenizeresults.out
#SBATCH -e outputs/tokenize.err

python3 run_tokenize.py