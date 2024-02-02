#!/bin/bash

IMAGES_DIR="/home/dhruv/Projects/imSitu_Clip/data/xtf_correct_imgs"
PYTHON_SCRIPT="inference.py"

cd /home/dhruv/Projects/CoFormer

for IMAGE_FILE in "$IMAGES_DIR"/*.jpg; do
    echo "Processing $IMAGE_FILE..."
    python "$PYTHON_SCRIPT" --image_path "$IMAGE_FILE" --saved_model "CoFormer/CoFormer_checkpoint.pth" --output_dir "/home/dhruv/Projects/imSitu_Clip/data/output/coformer_inference_xtf_correct"
done