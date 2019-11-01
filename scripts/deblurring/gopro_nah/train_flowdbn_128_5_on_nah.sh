#!/usr/bin/env bash
source ../../common.sh

# schedule if busy
wait-for-gpu

# datasets
# GOPRO_HOME=$DATASETS_HOME/DeepVideoDeblurring_Dataset/
GOPRO_HOME=$DATASETS_HOME/GOPRO_Large/


# model and checkpoint
MODEL=FlowDBN
CHECKPOINT=None

# Note that you need to specify a valid pwc checkpoint
PRETRAINED="$EXPERIMENTS_HOME/prewarping_networks/pwcnet.ckpt"


# LONG CONFIG
PREFIX="flowdbn128-5-nah"

TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$TIME-$PREFIX-$MODEL"
python ../../../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--loss=DBNLoss \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[308, 358, 408, 458, 508, 558]" \
--total_epochs=608 \
--model=$MODEL \
--model_pretrained_pwcnet=$PRETRAINED \
--optimizer=Adam \
--optimizer_lr=0.005 \
--prefix=$PREFIX \
--save=$SAVE_PATH \
--training_dataset=GoProNahTrain \
--training_dataset_sequence_length=5 \
--training_dataset_random_crop="[128, 128]" \
--training_dataset_num_samples_per_example=8 \
--training_dataset_num_workers=12 \
--training_dataset_num_examples=-1 \
--training_dataset_root=${GOPRO_HOME}