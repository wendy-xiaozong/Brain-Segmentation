#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=48000M  # memory
#SBATCH --output=tensorboard-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-06:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn gcc/8.3.0

SOURCEDIR=/home/jueqi/scratch

# Prepare virtualenv
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"
LOG_DIR=/home/jueqi/scratch/seg138_log

# run script
echo -e '\n\n\n'
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0