#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest
source deactivate
source activate llm_safety_39
cd /scratch/szinjad/llm-sensitivity
python3 src/scripts/generation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/perturbed/catHarmQA/catqa_char.csv --question_columns ocr_n1_question,keyboard_n1_question,random_insert_n1_question,random_substitute_n1_question,random_swap_n1_question,random_delete_n1_question,ocr_n2_question,keyboard_n2_question,random_insert_n2_question,random_substitute_n2_question,random_swap_n2_question,random_delete_n2_question,ocr_n3_question,keyboard_n3_question,random_insert_n3_question,random_substitute_n3_question,random_swap_n3_question,random_delete_n3_question,ocr_n4_question,keyboard_n4_question,random_insert_n4_question,random_substitute_n4_question,random_swap_n4_question,random_delete_n4_question,ocr_n5_question,keyboard_n5_question,random_insert_n5_question,random_substitute_n5_question,random_swap_n5_question,random_delete_n5_question