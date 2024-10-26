#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest
source deactivate
source activate llm_safety_39
cd /scratch/szinjad/llm-sensitivity
python3 src/scripts/perturbation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/original/catHarmQA/data_catqa_english.csv --perturbation_level word --perturbation_type synonym_wordnet,spelling,random_insert_cwe,random_substitute_cwe,random_swap_cwe,random_delete_cwe,random_insert_emb,random_substitute_emb,random_swap_emb,random_delete_emb --query_columns Question --max_perturb 5