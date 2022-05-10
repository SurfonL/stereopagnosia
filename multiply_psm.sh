#!/bin/bash
#SBATCH -J mul-psm13
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o ./results/%x_stdout_%j.txt
#SBATCH -e ./results/%x_stderr_%j.txt
#SBATCH --gres=gpu

python ./src/study_run.py