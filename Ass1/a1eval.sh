#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o resultse.out
#SBATCH -e resultse.err

python3 evala1.py