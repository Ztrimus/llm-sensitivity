#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/sbatch_log/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/sbatch_log/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest

source activate llm-safety

cd /scratch/szinjad/llm-sensitivity

python3 src/experiments/get_answer_from_models.py

