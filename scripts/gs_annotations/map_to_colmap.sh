#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE="$(pwd)"

# Activate conda environment
source .venv/bin/activate


xvfb-run -a  python preprocessing/gs_anno/map_to_colmap.py \
    --config "$VLSG_SPACE/preprocessing/gs_anno/gs_anno.yaml" \
    --output_dir "$DATA_ROOT_DIR/files/gs_annotations" \
    ${args[@]}
