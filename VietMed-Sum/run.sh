#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch

# List of checkpoints
checkpoints=(
#     "vinai/bartpho-syllable-base"
#     "vinai/bartpho-word-base"
#     "VietAI/vit5-base-vietnews-summarization"
    "VietAI/vit5-base"
#     "facebook/mbart-large-50"
#     "google/mt5-base"
#     "vinai/bartpho-syllable"
#     "vinai/bartpho-word"
#     "VietAI/vit5-large"
#     "VietAI/vit5-large-vietnews-summarization"
#     "google/mt5-large"
)

# Default parameters
epoch=100
lr=2e-5
batch_size=16
save_steps=1000
max_steps=5000

# GPU IDs
gpus=(0 1 2 3 4 5 6 7) # List of GPU IDs available for training

# Counter for GPUs
gpu_counter=0

# Loop through checkpoints and run the script for each one in parallel
for checkpoint in "${checkpoints[@]}"
do
    # Set GPU ID
    gpu_id=${gpus[gpu_counter]}

    # Check if the model is mT5 and set prefix accordingly
    if [[ $checkpoint == *"mt5"* ]]; then
        prefix="summarize: "
    else
        prefix=""
    fi
    echo $prefix
    echo "Running training for checkpoint: $checkpoint on GPU: $gpu_id with prefix: '$prefix'"
    CUDA_VISIBLE_DEVICES=$gpu_counter python train_sum.py --checkpoint "$checkpoint" --prefix "$prefix" --epoch $epoch --lr $lr --batch_size $batch_size --save_steps $save_steps --max_steps $max_steps #&
    
    # Increment GPU counter and reset if it exceeds the number of GPUs
    ((gpu_counter++))
    if [ $gpu_counter -ge ${#gpus[@]} ]; then
        gpu_counter=0
    fi

    # Optional: Wait a bit before starting the next process to ensure GPUs are not overloaded
#     sleep 5s
done

# Wait for all parallel jobs to finish
wait
echo "All training processes completed."
