#!/bin/bash
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -t 0-10:00
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
conda activate /net/rcstorenfs02/ifs/rc_labs/dominici_lab/lab/projects/pm-mortality-generative/ahmet/diffusers/conda_env

# Run program
accelerate launch modules/vqgan/train_vqgan.py run_id=$SLURM_JOB_ID
