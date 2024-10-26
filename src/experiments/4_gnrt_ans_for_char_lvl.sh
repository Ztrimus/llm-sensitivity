#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/sbatch_log/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/sbatch_log/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest
source deactivate
source activate llm_safety_39
cd /scratch/szinjad/llm-sensitivity
python3 src/scripts/generation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/original/catHarmQA/data_catqa_english.csv --question_columns Question