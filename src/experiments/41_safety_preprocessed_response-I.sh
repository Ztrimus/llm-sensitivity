#!/bin/bash
#SBATCH -N 1                          # Single node
#SBATCH --gres=gpu:a100:4             # 4 A100s per node
#SBATCH -c 8                          # 8 CPUs per GPU
#SBATCH --mem=320G                    # ~80GB per GPU
#SBATCH -p general
#SBATCH -q private
#SBATCH -t 01:00:00
#SBATCH -o /path/to/slurm.%j.out
#SBATCH -e /path/to/slurm.%j.err
#SBATCH --mail-type=ALL

# --- Core Modules ---
module load mamba/latest
module load cuda/12.2
source deactivate
source activate llm_safety_39

# --- Accelerate Configuration ---
export HF_HOME=/scratch/$USER/cache/huggingface
export CUDA_LAUNCH_BLOCKING=0         # Disable for performance
export TORCH_USE_CUDA_DSA=0           # Disable debug mode

# --- Job Execution ---
cd /scratch/szinjad/llm-sensitivity
export PYTHONPATH=$(pwd)/src

accelerate launch \
  --num_processes 4 \
  --mixed_precision fp16 \
  --multi_gpu \
  src/scripts/safety_prepro_res.py \
  --dataset_path /scratch/szinjad/llm-sensitivity/data/analyzed/catHarmQA/combined_catqa_sample.csv
