#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o outputs/resultse4.out
#SBATCH -e outputs/resultse4.err

python3 evala1.py