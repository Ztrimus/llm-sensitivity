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
python3 src/scripts/generation.py --model llama3 --dataset_path /scratch/szinjad/llm-sensitivity/data/perturbed/catHarmQA/catqa_word.csv --question_columns Question_word_random_substitute_cwe_n3,Question_word_random_insert_emb_n3,Question_word_random_substitute_emb_n3,Question_word_synonym_wordnet_n4,Question_word_spelling_n4,Question_word_random_insert_cwe_n4,Question_word_random_substitute_cwe_n4,Question_word_random_insert_emb_n4,Question_word_random_substitute_emb_n4,Question_word_synonym_wordnet_n5,Question_word_spelling_n5,Question_word_random_insert_cwe_n5,Question_word_random_substitute_cwe_n5,Question_word_random_insert_emb_n5,Question_word_random_substitute_emb_n5