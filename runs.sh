#!/bin/bash
#SBATCH --gres=gpu:p100:1  # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --output=data-path.out  # %N for node name, %j for jobID
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL


work_dir=$(pwd)  # /home/jueqi/projects/def-jlevman/jueqi/MRBrainS18/Code
MAIN_DIR=${work_dir%/*}
MAIN_DIR=${MAIN_DIR%/*}
DATA_DIR="$MAIN_DIR/MRBrainS18/Data/training" # The dataset folder
FILES_CHECKPOINT="$work_dir/run_$RUN/checkpoints/run_${RUN}_data.p"
TRAIN_SUBJECTS=5
MODEL="cnn_3d_1"
LOSS_TYPE="log_loss"
HUBER_DELTA=1
PATCH_SIZE="8,24,24"
BATCH_SIZE=64
KEEP_CHECKPOINT=6
MAX_STEPS=200000
STEPS_TO_VAL=2000
LEARNING_RATE=1e-4
LEARNING_RATE_DECREASE=0.1
STEPS_TO_LEARNING_RATE_UPDATE=15000
STEPS_TO_SAVE_CHECKPOINT=1000
CHECKPOINT_PATH="$work_dir/train/run_$RUN/checkpoints/run_${RUN}"

module load cuda cudnn
source /home/jueqi/tensorflow/bin/activate
mkdir "./run_${RUN}/checkpoints"
python ./utils/data.py \
    --data_dir="$DATA_DIR" \
    --files_checkpoint="$FILES_CHECKPOINT" \
    --train_subjects=$TRAIN_SUBJECTS \
    --model="$MODEL" \
    --loss_type="$LOSS_TYPE" \
    --huber_delta="$HUBER_DELTA" \
    --patch_size="$PATCH_SIZE" \
    --batch_size=$BATCH_SIZE \
    --keep_checkpoint=$KEEP_CHECKPOINT \
    --max_steps=$MAX_STEPS \
    --steps_to_val=$STEPS_TO_VAL \
    --learning_rate=$LEARNING_RATE \
    --learning_rate_decrease=$LEARNING_RATE_DECREASE \
    --steps_to_learning_rate_update=$STEPS_TO_LEARNING_RATE_UPDATE \
    --steps_to_save_checkpoint=$STEPS_TO_SAVE_CHECKPOINT \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --test_checkpoints="$TEST_CHECKPOINTS" \
    --cuda_device="$CUDA_DEVICE"