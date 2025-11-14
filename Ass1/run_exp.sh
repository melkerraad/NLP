#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o resultsd.out
#SBATCH -e resultsd.err

python3 explore_data.py