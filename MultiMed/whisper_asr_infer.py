import os
import re
import string
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

language = 'English'
temp_ckpt_folder = 'whisper/temp'
ckpt_dir = f'whisper-small-english/checkpoint-4000*'

os.makedirs(temp_ckpt_folder, exist_ok=True)
ckpt_dir_parent = str(Path(ckpt_dir).parent)

files_to_copy = [
    "added_tokens.json", "normalizer.json", "preprocessor_config.json",
    "special_tokens_map.json", "generation_config.json", "tokenizer_config.json",
    "merges.txt", "vocab.json", "config.json", "model.safetensors", "training_args.bin"
]
os.system(f"cp {' '.join([f'{ckpt_dir_parent}/{file}' for file in files_to_copy])} {temp_ckpt_folder}")
model_id = temp_ckpt_folder

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
)

dataset = load_dataset('wnkh/MultiMed', language, split='corrected.test')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def normalize_text_to_chars(text):
    """Removes punctuation and extra spaces from text."""
    if isinstance(text, str):
        return " ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", text).split())
    return text

def transcribe_audio(batch):
    """Transcribes audio using the Whisper model."""
    batch["prediction"] = pipe(batch["audio"], return_timestamps=True)['text']
    return batch

def infer_and_save_to_csv(dataset):
    """Processes the dataset, transcribes audio, and saves predictions to a CSV file."""
    predictions = [transcribe_audio(batch) for batch in tqdm(dataset, desc="Transcribing")]
    
    df = pd.DataFrame({
        "ID": [f"sentence_{i+1}" for i in range(len(predictions))],
        "Original Text": [batch["text"] for batch in predictions],
        "Prediction": [batch["prediction"] for batch in predictions],
        "French": [batch.get("French", "") for batch in predictions],
        "Chinese": [batch.get("Chinese", "") for batch in predictions],
        "German": [batch.get("German", "") for batch in predictions],
        "Vietnamese": [batch.get("Vietnamese", "") for batch in predictions]
    })
    
    output_file = f"Whisper-small/Test/{language}_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Predictions and corresponding texts saved to {output_file}")
    
infer_and_save_to_csv(dataset)