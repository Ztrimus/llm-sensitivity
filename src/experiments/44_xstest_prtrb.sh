#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest
source deactivate
source activate llm_safety_39
cd /scratch/szinjad/llm-sensitivity
export PYTHONPATH=$(pwd)/src
python3 src/scripts/perturbation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/original/XSTest/xstest.csv --perturbation_level char --perturbation_type ocr,keyboard,random_insert,random_substitute,random_swap,random_delete --max_perturb 5 --query_columns prompt
python3 src/scripts/perturbation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/original/XSTest/xstest.csv --perturbation_level word --perturbation_type synonym_wordnet,spelling,random_insert_cwe,random_substitute_cwe,random_insert_emb,random_substitute_emb --max_perturb 5 --query_columns prompt
python3 src/scripts/perturbation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/original/XSTest/xstest.csv --perturbation_level sntnc --perturbation_type bck_trnsltn,paraphrase --max_perturb 1 --query_columns prompt