#!/bin/bash

args=("$@")

# Set environment variables
export VLSG_SPACE="$(pwd)"
# get output directory argument if it exists
for i in "$@"
do
case $i in
    -o=*|--output_dir=*)
    export VLSG_TRAINING_OUT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Set output directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export VLSG_TRAINING_OUT_DIR="$SCRATCH/training_recon_scene/$timestamp"

# Initialize conda and activate the environment
source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# Run training script
python src/trainval/train_reconstruction.py --config scripts/train_val/train.yaml --log_steps 1 output_dir=\"$VLSG_TRAINING_OUT_DIR\" ${args[@]}
