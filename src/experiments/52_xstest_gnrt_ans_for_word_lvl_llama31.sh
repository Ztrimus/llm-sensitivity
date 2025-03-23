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
python3 src/scripts/generation.py --model llama31 --dataset_path /scratch/szinjad/llm-sensitivity/data/perturbed/xstest/xstest_word.csv --question_columns prompt_word_synonym_wordnet_n1,prompt_word_spelling_n1,prompt_word_random_insert_cwe_n1,prompt_word_random_substitute_cwe_n1,prompt_word_random_insert_emb_n1,prompt_word_random_substitute_emb_n1,prompt_word_synonym_wordnet_n2,prompt_word_spelling_n2,prompt_word_random_insert_cwe_n2,prompt_word_random_substitute_cwe_n2,prompt_word_random_insert_emb_n2,prompt_word_random_substitute_emb_n2,prompt_word_synonym_wordnet_n3,prompt_word_spelling_n3,prompt_word_random_insert_cwe_n3,prompt_word_random_substitute_cwe_n3,prompt_word_random_insert_emb_n3,prompt_word_random_substitute_emb_n3,prompt_word_synonym_wordnet_n4,prompt_word_spelling_n4,prompt_word_random_insert_cwe_n4,prompt_word_random_substitute_cwe_n4,prompt_word_random_insert_emb_n4,prompt_word_random_substitute_emb_n4,prompt_word_synonym_wordnet_n5,prompt_word_spelling_n5,prompt_word_random_insert_cwe_n5,prompt_word_random_substitute_cwe_n5,prompt_word_random_insert_emb_n5,prompt_word_random_substitute_emb_n5