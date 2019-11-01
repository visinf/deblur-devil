#!/usr/bin/env bash
source ../common.sh

# schedule if busy
wait-for-gpu

# datasets
MNIST_HOME=${DATASETS_HOME}/mnist

# model and checkpoint
MODEL=LeNet
CHECKPOINT=None

# save path
PREFIX="simple"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$TIME-$PREFIX-$MODEL"

# training configuration
python ../../main.py \
--batch_size=64 \
--checkpoint=$CHECKPOINT \
--clip_grad=0 \
--loss=ClassificationLoss \
--logging_loss_graph=True \
--logging_model_graph=True \
--model=$MODEL \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[5, 10, 15]" \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_group="{'params': '*bias*', 'lr': 0.01, 'weight_decay': 0}" \
--prefix=$PREFIX \
--loshchilov_weight_decay=0.01 \
--save="$SAVE_PATH" \
--total_epochs=20 \
--training_dataset=MnistTrain \
--training_dataset_root=$MNIST_HOME \
--training_key=xe \
--validation_dataset=MnistValid \
--validation_dataset_root=$MNIST_HOME \
--validation_keys=top1,xe \
--validation_modes=max,min