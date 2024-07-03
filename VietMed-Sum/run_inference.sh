#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch

# Path to the base directory
BASE_DIR=~/../../scratch/knguyen07/new_models

ls "$BASE_DIR"
# Path to the inference.py script
INFERENCE_SCRIPT_PATH="inference.py"

# Loop through each directory in the base directory
# for dir in $(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
# do
#     echo "Running inference on directory: $dir"
#     CUDA_VISIBLE_DEVICES=0 python "$INFERENCE_SCRIPT_PATH" --path "$dir"
# done
for dir in $(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
do
    # Look for directories inside the current directory that start with "checkpoint-"
    for subdir in $(find "$dir" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*")
    do
        # Extract the name of the subdirectory without the path
        subdirname=$(basename "$subdir")
        dirname=$(basename "$dir")
        # Check if the subdirectory name does not end with "_100"
        if [[ ! $dirname == *_100 ]]; then
            echo "Running inference on directory: $subdir"
            CUDA_VISIBLE_DEVICES=0 python "$INFERENCE_SCRIPT_PATH" --path "$subdir"
        fi
    done
done
