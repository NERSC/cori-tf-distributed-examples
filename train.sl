#!/bin/bash -l

#SBATCH -t 10
#SBATCH -p regular
#SBATCH -C haswell
#SBATCH --qos=premium
rm -rf ./logs/*
module load deeplearning
srun python -u main.py
