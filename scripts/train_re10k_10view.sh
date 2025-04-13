#!/bin/bash
#SBATCH --job-name=re10k_10view
#SBATCH --time=2-00:00:00              # Time limit: 2 days
#SBATCH --gpus=a100_80gb:3         # Request 4 A100 80GB GPUs
#SBATCH --ntasks=1                          # Number of tasks (processes) - changed to 1
#SBATCH --cpus-per-task=10                         # Number of CPU cores per task - added this line
#SBATCH --mem-per-cpu=20000           # Memory per CPU in MB (20GB)

cd "/cluster/home/tgueloglu/3DV/UnposedSplat"
source ~/.bashrc   
conda activate noposplat

module load stack/2024-06 cuda/12.4.1

python -m src.main +experiment=re10k_10view wandb.mode=online wandb.name=re10k_10view