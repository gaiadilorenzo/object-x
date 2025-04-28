#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

source .venv/bin/activate

python preprocessing/voxel_anno/voxelise_features.py \
    --config "preprocessing/voxel_anno/voxel_anno.yaml" \
    --model_dir "$DATA_ROOT_DIR" \
    ${args[@]}
