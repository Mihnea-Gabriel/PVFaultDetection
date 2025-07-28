#!/usr/bin/env bash
# run.sh - Unified interface for training, testing, and demo of the Solar CV project
# Usage: ./run.sh <action> [options]

set -euo pipefail

# Default parameters
DATA_DIR="datasets"
MODEL_NAME="resnet50"
BATCH_SIZE=16
EPOCHS=8
LR=3e-4
IMG_HEIGHT=288
IMG_WIDTH=288
TRAIN_SPLIT=0.7
VAL_SPLIT=0.15
TEST_SPLIT=0.15
NUM_WORKERS=8
SEED=42
DEVICE="cuda"
PATIENCE=10

# For test/demo actions
TRAINED_MODEL=""
VIDEO_PATH=""

# Print general help
show_general_help() {
  cat <<EOF
Usage: $(basename "$0") <action> [options]

Available actions:
  train      Train a new model
  test       Evaluate a saved model
  demo       Run the demo on a video file

Run './run.sh <action> -h' to see options for that action.
EOF
}

# Help for train
show_train_help() {
  cat <<EOF
train action options:
  --data-dir DIR        Path to dataset (default: $DATA_DIR)
  --model-name NAME     Model backbone (default: $MODEL_NAME)
  --batch-size N        Batch size (default: $BATCH_SIZE)
  --epochs N            Number of epochs (default: $EPOCHS)
  --lr LR               Learning rate (default: $LR)
  --img-size H W        Image size height and width (default: $IMG_HEIGHT $IMG_WIDTH)
  --train-split F       Fraction for training set (default: $TRAIN_SPLIT)
  --val-split F         Fraction for validation set (default: $VAL_SPLIT)
  --test-split F        Fraction for test set (default: $TEST_SPLIT)
  --num-workers N       DataLoader workers (default: $NUM_WORKERS)
  --seed N              RNG seed (default: $SEED)
  --device DEVICE       Compute device (default: $DEVICE)
  --patience N          Early stopping patience (default: $PATIENCE)
EOF
}

# Help for test
show_test_help() {
  cat <<EOF
test action options:
  --trained-model PATH  Path to trained checkpoint (required)
  --data-dir DIR        Path to dataset (default: $DATA_DIR)
  --model-name NAME     Model backbone (default: $MODEL_NAME)
  --device DEVICE       Compute device (default: $DEVICE)
  # Other flags from 'train' are also available
EOF
}

# Help for demo
show_demo_help() {
  cat <<EOF
demo action options:
  --video PATH          Path to input video file (required)
  --trained-model PATH  Path to trained checkpoint (required for demo)
  --device DEVICE       Compute device for demo (default: $DEVICE)
  --imgsz N             Inference resolution (default: $IMG_HEIGHT)
EOF
}

# Parse action
if [[ $# -lt 1 ]]; then
  show_general_help
  exit 1
fi
ACTION="$1"; shift

# Show action-specific help
if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  case "$ACTION" in
    train) show_train_help ;; 
    test)  show_test_help  ;; 
    demo)  show_demo_help  ;; 
    *)     show_general_help ;; 
  esac
  exit 0
fi

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)   DATA_DIR="$2"; shift 2 ;; 
    --model-name) MODEL_NAME="$2"; shift 2 ;; 
    --batch-size) BATCH_SIZE="$2"; shift 2 ;; 
    --epochs)     EPOCHS="$2"; shift 2 ;; 
    --lr)         LR="$2"; shift 2 ;; 
    --img-size)   IMG_HEIGHT="$2"; IMG_WIDTH="$3"; shift 3 ;; 
    --train-split) TRAIN_SPLIT="$2"; shift 2 ;; 
    --val-split)  VAL_SPLIT="$2"; shift 2 ;; 
    --test-split) TEST_SPLIT="$2"; shift 2 ;; 
    --num-workers) NUM_WORKERS="$2"; shift 2 ;; 
    --seed)       SEED="$2"; shift 2 ;; 
    --device)     DEVICE="$2"; shift 2 ;; 
    --patience)   PATIENCE="$2"; shift 2 ;; 
    --trained-model) TRAINED_MODEL="$2"; shift 2 ;; 
    --video)      VIDEO_PATH="$2"; shift 2 ;; 
    --imgsz)      IMG_HEIGHT="$2"; shift 2 ;; 
    *) echo "Unknown option: $1"; exit 1 ;; 
  esac
done

# Actions
train() {
  echo "➡️ Starting training with model=$MODEL_NAME on device=$DEVICE"
  python src/train.py \
    --data-dir "$DATA_DIR" \
    --model-name "$MODEL_NAME" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --img-size "$IMG_HEIGHT" "$IMG_WIDTH" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --test-split "$TEST_SPLIT" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --patience "$PATIENCE"
}

test_model() {
  if [[ -z "$TRAINED_MODEL" ]]; then
    echo "❗ Error: --trained-model must be provided for 'test'"
    exit 1
  fi
  echo "➡️ Evaluating checkpoint=$TRAINED_MODEL on device=$DEVICE"
  python src/train.py \
    --data-dir "$DATA_DIR" \
    --model-name "$MODEL_NAME" \
    --testing 1 \
    --trained-model "$TRAINED_MODEL" \
    --device "$DEVICE"
}

demo() {
  if [[ -z "$VIDEO_PATH" || -z "$TRAINED_MODEL" ]]; then
    echo "❗ Error: --video and --trained-model must be provided for 'demo'"
    exit 1
  fi
  echo "➡️ Running demo with checkpoint=$TRAINED_MODEL on video=$VIDEO_PATH"
  python src/utils/demo.py \
    --weights "$TRAINED_MODEL" \
    --video "$VIDEO_PATH" \
    --device "$DEVICE" \
    --imgsz "$IMG_HEIGHT"
}

# Dispatch
case "$ACTION" in
  train) train ;; 
  test)  test_model ;; 
  demo)  demo ;; 
  *)
    echo "Unknown action: $ACTION"
    show_general_help
    exit 1
    ;; 
esac

