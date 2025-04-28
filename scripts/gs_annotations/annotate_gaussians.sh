#!/bin/bash

args=("$@")

# Environment setup
export VLSG_SPACE="$(pwd)"
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# Activate conda environment
source .venv/bin/activate

iterations=7000
densify_until_iter=15_000


python preprocessing/gs_anno/annotate_gaussians.py \
    --densify_until_iter $densify_until_iter \
    --iterations $iterations \
    --save_iterations $iterations \
    --test_iterations $iterations \
    --config "$VLSG_SPACE/preprocessing/gs_anno/gs_anno.yaml" \
    --source_dir "$DATA_ROOT_DIR" \
    --model_dir "$DATA_ROOT_DIR" \
    ${args[@]}
