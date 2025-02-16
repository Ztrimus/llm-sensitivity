#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p general
#SBATCH -q private
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80G
#SBATCH -o /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.out
#SBATCH -e /scratch/szinjad/llm-sensitivity/supports/job_logs/slurm.%j.err
#SBATCH --mail-type=ALL

module load mamba/latest
source deactivate
source activate llm_safety_39
cd /scratch/szinjad/llm-sensitivity
export PYTHONPATH=$(pwd)/src
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_LAUNCH_BLOCKING=1
python3 src/scripts/safety_prepro_res.py --dataset_path /scratch/szinjad/llm-sensitivity/data/analyzed/catHarmQA/combined_catqa_sample.csv