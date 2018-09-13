#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1000
#SBATCH --time=24:00:00
#SBATCH --partition=skylake

module load gcc/5.5.0
module load openmpi/3.0.0
module load cuda/9.0.176

srun ./SPEO_GPU -dim 100 -island_size $1 -total_functions $2-$3 -total_runs 1-15 -regroup_option dynamic_and_random
