#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o ./run_logs/pytorch_%j.out 
#SBATCH -e ./run_logs/pytorch_%j.err 

# Load software modules and source conda environment
module load python/3.10.13-fasrc01
conda deactivate
conda deactivate
conda deactivate
source activate try2

# Run program
python preprocessing/vae_hf.py
