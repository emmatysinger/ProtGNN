#!/bin/bash
#SBATCH --job-name=esm_training
#SBATCH --output=%x.out
#SBATCH --error=%x.log
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G

export JOB_TYPE=finetune

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
python TxGNN/training_script.py -f True 

#cp $SLURM_JOB_NAME.out TxGNN/training_logs/
#python TxGNN/training_logs/parser.py -i TxGNN/training_logs/$SLURM_JOB_NAME.out -o TxGNN/training_logs/$SLURM_JOB_NAME.csv -t $JOB_TYPE
#python TxGNN/training_logs/eval.py -i TxGNN/training_logs/$SLURM_JOB_NAME.csv -s TxGNN/training_logs/$SLURM_JOB_NAME -t $JOB_TYPE

