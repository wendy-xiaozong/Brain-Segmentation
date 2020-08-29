#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=process-%j.out  # %N for node name, %j for jobID
#SBATCH --time=01-00:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn gcc/8.3.0

SOURCEDIR=/home/jueqi/scratch

# Prepare virtualenv
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

#echo "$(date +"%T"):  start to install wheels"
#pip install -r $SOURCEDIR/requirements.txt && echo "$(date +"%T"):  install successfully!"

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo

echo -e '\n'
cd $SLURM_TMPDIR
mkdir work

# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/scratch/Data/readable_data.tar -C work && echo "$(date +"%T"):  Copied data"
# tar -xf /home/jueqi/scratch/Data/processed_ADNI.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...

cd work
# tar -tvf .tar | grep ^d
## The computations are done, so clean up the data set...
# avoiding batch_size < gpus number

# run script
echo -e '\n\n\n'
python3 /home/jueqi/scratch/Unet_seg138_13/cropping.py

tar -cf /home/jueqi/scratch/Data/cropped_ADNI.tar cropped_img/ cropped_label/
tar -cf /home/jueqi/scratch/Data/strange_ADNI.tar strange_img/ strange_label/
#python3 /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/data/const.py