#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 4:00:00
#SBATCH -G a100:1
#SBATCH --mem=80G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest

source activate llm-safety

cd /scratch/szinjad/llm-sensitivity

python3 src/experiments/get_answer_from_models.py

