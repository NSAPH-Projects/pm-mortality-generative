#!/bin/bash
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o ./run_logs/pytorch_%j.out 
#SBATCH -e ./run_logs/pytorch_%j.err 

# Conda environment location
CONDA_ENV="/net/rcstorenfs02/ifs/rc_labs/dominici_lab/lab/projects/pm-mortality-generative/ahmet/diffusers/conda_env"
TRAINING_SCRIPT="src/train_vqgan.py"

# Load software modules and source conda environment
module load python/3.10.13-fasrc01
conda deactivate
conda activate $CONDA_ENV

# Run program
accelerate launch --config_file conf/accelerate/cannon-no-opt.yaml $TRAINING_SCRIPT run_id=$SLURM_JOB_ID
