#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4  # on Cedar
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M  # memory
#SBATCH --output=seg138-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-06:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
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

# force to synchronization, can pinpoint the exact number of lines of error code where our memory operation is observed
CUDA_LAUNCH_BLOCKING=1

echo -e '\n'
cd $SLURM_TMPDIR
mkdir work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
#tar -xf /home/jueqi/scratch/Data/readable_data.tar -C work && echo "$(date +"%T"):  Copied data"
tar -xf /home/jueqi/scratch/Data/cropped_resampled_ADNI.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...

cd work

BATCH_SIZE=4
NODES=1
GPUS=4
OUT_CHANNELS_FIRST_LAYER=32
LEARNING_RATE=1e-4
KERNEL_SIZE=3
DEEPTH=5
PATCH_SIZE=128
# MODEL=unet
MODEL=highResNet
# only to avoid the border effect.
PATCH_OVERLAP=4
RUN=40
LOG_DIR=/home/jueqi/scratch/seg138_log

# run script
echo -e '\n\n\n'
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/jueqi/scratch/Unet_seg138_9/Lit_train.py \
       --gpus=$GPUS \
       --batch_size=$BATCH_SIZE \
       --nodes=$NODES \
       --name="score and loss, include_background=True, dice loss, one more bottom layer, and enable visulization" \
       --TensorBoardLogger="$LOG_DIR" \
       --model="$MODEL" \
       --learning_rate=$LEARNING_RATE \
       --out_channels_first_layer=$OUT_CHANNELS_FIRST_LAYER \
       --run=$RUN \
       --deepth=$DEEPTH \
       --kernel_size=$KERNEL_SIZE \
       --patch_size=$PATCH_SIZE \
       --patch_overlap=$PATCH_OVERLAP \
       --include_background \
       --checkpoint_file="epoch=3-val_dice=0.65432.ckpt"
#        --fast_dev_run



#python3 /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/data/const.py