#!/usr/bin/env bash
source ../../common.sh

# schedule if busy
wait-for-gpu

# datasets
GOPRO_HOME="$DATASETS_HOME/DeepVideoDeblurring_Dataset/"


# model and checkpoint
MODEL=FlowDBN
CHECKPOINT=None

# Note that you need to specify a valid pwc checkpoint
PRETRAINED="$EXPERIMENTS_HOME/prewarping_networks/pwcnet.ckpt"

# LONG CONFIG
PREFIX="flowdbn-192-7-su"

TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$TIME-$PREFIX-$MODEL"
python ../../../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--loss=DBNLoss \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[108, 126, 144, 162, 180, 198]" \
--total_epochs=216 \
--model=$MODEL \
--model_pretrained_pwcnet=$PRETRAINED \
--optimizer=Adam \
--optimizer_lr=0.005 \
--prefix=$PREFIX \
--save=$SAVE_PATH \
--training_dataset=GoProSuQuantitativeTrain \
--training_dataset_sequence_length=7 \
--training_dataset_random_crop="[192, 192]" \
--training_dataset_num_samples_per_example=8 \
--training_dataset_num_workers=12 \
--training_dataset_num_examples=-1 \
--training_dataset_root=${GOPRO_HOME}