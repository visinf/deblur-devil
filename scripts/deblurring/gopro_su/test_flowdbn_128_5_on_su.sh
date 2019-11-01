#!/usr/bin/env bash
source ../../common.sh

# schedule if busy
wait-for-gpu

# datasets
GOPRO_HOME="$DATASETS_HOME/DeepVideoDeblurring_Dataset/"


# model and checkpoint
MODEL=FlowDBN
CHECKPOINT="$EXPERIMENTS_HOME/path/to/checkpoint.ckpt"

# Note that you need to specify a valid pwc checkpoint
PRETRAINED="$EXPERIMENTS_HOME/prewarping_networks/pwcnet.ckpt"

# LONG CONFIG
PREFIX="flowdbn-128-5-su"

TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$TIME-$PREFIX-$MODEL-inference"
python ../../../main.py \
--batch_size=1 \
--checkpoint=$CHECKPOINT \
--loss=DBNLoss \
--model=$MODEL \
--model_pretrained_pwcnet=$PRETRAINED \
--prefix=$PREFIX \
--save=$SAVE_PATH \
--validation_dataset=GoProSuQuantitativeValid \
--validation_dataset_sequence_length=5 \
--validation_dataset_num_workers=12 \
--validation_dataset_root=${GOPRO_HOME} \
--validation_keys=psnr \
--validation_modes=max \
--visualizer=GoProInference \
--visualizer_save="$SAVE_PATH"/inference
