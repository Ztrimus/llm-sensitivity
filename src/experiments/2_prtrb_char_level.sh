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
python3 src/scripts/perturbation.py \
    --dataset_path /scratch/szinjad/llm-sensitivity/data/original/catHarmQA/data_catqa_english.csv \
    --perturbation_level char \
    --perturbation_type ocr \
    --max_char_perturb 5 \
    --columns Question 