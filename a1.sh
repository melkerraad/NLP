#!/bin/bash
#SBATCH --gres=gpu:L4:3
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -o results.out
#SBATCH -e results.err
#SBATCH -p long

python3 A1.py
