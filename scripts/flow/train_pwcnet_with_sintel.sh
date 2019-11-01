#!/usr/bin/env bash
source ../common.sh

# schedule if busy
wait-for-gpu

# datasets
FLYINGCHAIRS_HOME=${DATASETS_HOME}/FlyingChairs_release/data/
SINTEL_HOME=$DATASETS_HOME/MPI-Sintel-complete/

# model and checkpoint
MODEL=PWCNet
CHECKPOINT=None

# save path
PREFIX="train-flyingchairs-with-sintel"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/release/$TIME-$PREFIX-$MODEL"

# training configuration
python ../../main.py \
--batch_size=8 \
--checkpoint=${CHECKPOINT} \
--loss=MultiScaleEPE \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[108, 144, 180]" \
--model=${MODEL} \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--prefix=${PREFIX} \
--save="${SAVE_PATH}" \
--total_epochs=216 \
--training_dataset=FlyingChairsTrain \
--training_augmentation=RandomAffineFlow \
--training_dataset_num_examples=-1 \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=${FLYINGCHAIRS_HOME} \
--training_dataset_num_examples=-1 \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_dataset_num_workers=4 \
--validation_keys=epe \
--visualizer=FlowVisualizer