#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q grp_corman
#SBATCH -t 12:30:00
#SBATCH -G a100:2
#SBATCH --mem=80G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest

source activate explanation

cd ~/projects/Explanation-Eval/llama-exp/

python3 llama2_13b_bnb.py

