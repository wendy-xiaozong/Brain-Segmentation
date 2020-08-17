#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1  # on Cedar
#SBATCH --ntasks-per-node=2  # try again
#SBATCH --cpus-per-task=8  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M  # memory
#SBATCH --output=seg138-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-00:30      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn gcc/8.3.0

SOURCEDIR=/home/jueqi/scratch

# Prepare virtualenv
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

python3 /home/jueqi/scratch/Unet_seg138_4/summary_of_all_layers.py