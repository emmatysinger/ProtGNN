#!/bin/bash
#SBATCH --job-name=esm-pretrain
#SBATCH --output=finetuning_esm.out
#SBATCH --error=finetuning_esm.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

source /etc/profile.d/modules.sh
cat .bashrc
module load openmind8/cuda/11.7
module load openmind8/cudnn/8.7.0-cuda11

export PATH='/om/user/tysinger/anaconda3/bin':$PATH
source /om/user/tysinger/anaconda3/etc/profile.d/conda.sh

echo Activating txgnn_env2 ... 
conda activate txgnn_env2  # Activate your Python virtual environment if you have one

export PATH='~/.conda/envs/txgnn_env2/bin':$PATH

echo Starting training script ... 
python TxGNN/training_script.py
