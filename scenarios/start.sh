#!/bin/bash
#SBATCH --job-name=VisualImag
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=4
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/dinardo/.bashrc
conda activate tf2

python unsupervised_and_clustering.py --settings unsup_clust.yaml
