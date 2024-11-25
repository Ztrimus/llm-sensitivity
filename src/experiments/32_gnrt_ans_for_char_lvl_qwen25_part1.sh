/*
 * -----------------------------------------------------------------------
 * File: experiments/8_gnrt_ans_for_char_lvl_llama31_part1 copy.sh
 * Creation Time: Nov 23rd 2024, 3:05 pm
 * Author: Saurabh Zinjad
 * Developer Email: saurabhzinjad@gmail.com
 * Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
 * -----------------------------------------------------------------------
 */

#!/bin/bash

#SBATCH -N 1
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 15:00:00
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
python3 src/scripts/generation.py --model qwen25 --dataset_path /scratch/szinjad/llm-sensitivity/data/perturbed/catHarmQA/catqa_char.csv --question_columns Question_char_ocr_n1,Question_char_keyboard_n1,Question_char_random_insert_n1,Question_char_random_substitute_n1,Question_char_random_swap_n1,Question_char_random_delete_n1,Question_char_ocr_n2,Question_char_keyboard_n2,Question_char_random_insert_n2,Question_char_random_substitute_n2,Question_char_random_swap_n2,Question_char_random_delete_n2,Question_char_ocr_n3,Question_char_keyboard_n3,Question_char_random_insert_n3