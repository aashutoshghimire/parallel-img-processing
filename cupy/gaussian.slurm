#!/bin/bash
#SBATCH --job-name=gaussian            #for tracking job name
#SBATCH --time=15                  #amount of time requested (job will be killed if it runs longer than this)
#SBATCH --partition=p100      #Partitions determine what hardware you request, a100 gpu is faster
#SBATCH --gres=gpu:1                #for requesting a GPU (p100 has x8 in one node, a100 only has 1gpu)
#SBATCH --output=/home/%u/ceg7370/gaussian/%x-%u-%j.out    #Send output to ~/

echo "STARTING JOB"
srun singularity exec --nv /home/containers/cupy.sif python -u ./gaussian.py $1
echo "JOB FINISHED"
