#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 20:00:00
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
python3 src/scripts/generation.py --dataset_path /scratch/szinjad/llm-sensitivity/data/perturbed/catHarmQA/catqa_char.csv --model mistral --question_columns Question_char_ocr_n1,Question_char_keyboard_n1,Question_char_random_insert_n1,Question_char_random_substitute_n1,Question_char_random_swap_n1,Question_char_random_delete_n1,Question_char_ocr_n2,Question_char_keyboard_n2,Question_char_random_insert_n2,Question_char_random_substitute_n2,Question_char_random_swap_n2,Question_char_random_delete_n2,Question_char_ocr_n3,Question_char_keyboard_n3,Question_char_random_insert_n3,Question_char_random_substitute_n3,Question_char_random_swap_n3,Question_char_random_delete_n3,Question_char_ocr_n4,Question_char_keyboard_n4,Question_char_random_insert_n4,Question_char_random_substitute_n4,Question_char_random_swap_n4,Question_char_random_delete_n4,Question_char_ocr_n5,Question_char_keyboard_n5,Question_char_random_insert_n5,Question_char_random_substitute_n5,Question_char_random_swap_n5,Question_char_random_delete_n5