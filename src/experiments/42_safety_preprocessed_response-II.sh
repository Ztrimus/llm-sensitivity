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
python3 src/scripts/safety_prepro_res-simple.py --dataset_path /scratch/szinjad/llm-sensitivity/data/analyzed/catHarmQA/combined_catqa.csv