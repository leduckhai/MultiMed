declare -A LANGUAGE_CODES
LANGUAGE_CODES=(
    ["English"]="en"
    ["Vietnamese"]="vi"
    ["French"]="fr"
    ["German"]="de"
    ["Chinese"]="zh"
)

LANGUAGES=("English" "Vietnamese" "French" "German" "Chinese")

MODEL_NAME="openai/whisper-small"
SAMPLING_RATE=16000
NUM_PROC=2
TRAIN_STRATEGY="steps"
LEARNING_RATE=1.75e-5
WARMUP=20000
TRAIN_BATCHSIZE=48
EVAL_BATCHSIZE=32
NUM_STEPS=100000
NUM_EPOCHS=20
OUTPUT_BASE_DIR="output_models"
TRAIN_DATASETS="wnkh/MultiMed"
EVAL_DATASETS="wnkh/MultiMed"

mkdir -p "$OUTPUT_BASE_DIR"

for LANGUAGE in "${LANGUAGES[@]}"; do
    LANG_CODE=${LANGUAGE_CODES[$LANGUAGE]}
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/whisper-${LANG_CODE}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory $OUTPUT_DIR already exists. Skipping training for $LANGUAGE."
        continue
    fi
    
    echo "Starting training for language: $LANGUAGE ($LANG_CODE)"
    echo "Output directory: $OUTPUT_DIR"
    
    python train_asr.py \
        --model_name "$MODEL_NAME" \
        --language "$LANGUAGE" \
        --sampling_rate $SAMPLING_RATE \
        --num_proc $NUM_PROC \
        --train_strategy "$TRAIN_STRATEGY" \
        --learning_rate $LEARNING_RATE \
        --warmup $WARMUP \
        --train_batchsize $TRAIN_BATCHSIZE \
        --eval_batchsize $EVAL_BATCHSIZE \
        --num_steps $NUM_STEPS \
        --num_epochs $NUM_EPOCHS \
        --output_dir "$OUTPUT_DIR" \
        --train_datasets "$TRAIN_DATASETS" \
        --eval_datasets "$EVAL_DATASETS"
        
    echo "Finished training for language: $LANGUAGE ($LANG_CODE)"
    echo "-------------------------------------------------------"
done

echo "All training completed!"