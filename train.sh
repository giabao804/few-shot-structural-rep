#!/bin/bash

usage() {
    echo "Usage: $0 <mode>"
    echo "  <mode>    Training mode: 1 for 1-shot or 5 for 5-shot"
    echo
    echo "Examples:"
    echo "  bash $0 1    # Runs 1-shot training"
    echo "  bash $0 5    # Runs 5-shot training"
    exit 1
}


if [ "$#" -ne 1 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

MODE=$1

# Config parameters
DATASET="CWRU"
TRAINING_SAMPLES="60"
MODEL_NAME="few-shot-structural-rep"


if [ "$MODE" -eq 1 ]; then
    TRAIN_SCRIPT="train_1shot.py"
    TRAINING_SAMPLES_OPTION="--training_samples_CWRU $TRAINING_SAMPLES"
elif [ "$MODE" -eq 5 ]; then
    TRAIN_SCRIPT="train_5shot.py"
    TRAINING_SAMPLES_OPTION="--training_samples_CWRU $TRAINING_SAMPLES"
else
    echo "Error: Mode must be either 1 (1-shot) or 5 (5-shot)."
    usage
fi


if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found in the current directory."
    exit 1
fi


echo "========================================="
echo "Training Configuration:"
echo "Mode             : ${MODE}-shot"
echo "Dataset          : $DATASET"
echo "Training Samples : $TRAINING_SAMPLES"
echo "Model Name       : $MODEL_NAME"
echo "Training Script  : $TRAIN_SCRIPT"
echo "========================================="


echo "Starting ${MODE}-shot training..."
python3 "$TRAIN_SCRIPT" --dataset "$DATASET" $TRAINING_SAMPLES_OPTION --model_name "$MODEL_NAME"
echo "Training completed successfully for ${TRAINING_SAMPLES} training samples with ${MODE}-shot learning."
