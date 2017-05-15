#!/bin/bash -l

#SBATCH -t 10
#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -o batch_outputs/slurm_%N.%j.out
#SBATCH -e batch_outputs/slurm_%N.%j.out
#SBATCH --qos=premium
rm -rf ./logs/*
module load deeplearning
srun python -u main.py
