#!/bin/bash -l

#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -o batch_outputs/slurm_%N.%j.out
#SBATCH -e batch_outputs/slurm_%N.%j.out
#SBATCH --qos=premium
module load deeplearning
python get_data/download_and_convert_cifar_10.py
srun python -u main.py $@
