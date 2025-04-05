import pandas as pd
import re
import string
from evaluate import load

def normalize_text(text):
    """Normalize text by removing punctuation and spacing out characters."""
    punctuation_to_remove_regex = f"[{re.escape(string.punctuation)}]"
    text = re.sub(punctuation_to_remove_regex, " ", text).strip() if isinstance(text, str) else text
    return " ".join(text) if isinstance(text, str) else text

def compute_asr_scores(reference_texts, predicted_texts):
    """Compute CER and WER between references and predictions."""
    cer_metric = load("cer")
    wer_metric = load("wer")

    references = [normalize_text(text) for text in reference_texts]
    predictions = [normalize_text(text) for text in predicted_texts]

    cer_score = cer_metric.compute(references=references, predictions=predictions)
    wer_score = wer_metric.compute(references=references, predictions=predictions)

    return cer_score, wer_score

# === Load Predictions and Ground Truth ===
lang = "English"
e_set = "Test"

predict_path = f'Whisper-small/{e_set}/{lang}_predictions.csv'
groundtruth_path = f'multimed/{e_set}/{lang}.csv'

predict_df = pd.read_csv(predict_path)
label_df = pd.read_csv(groundtruth_path)

merged_df = pd.merge(label_df, predict_df, on="path", suffixes=('_label', '_predict'))

references = merged_df['Original Text'].tolist()
predictions = merged_df['Prediction'].tolist()

cer, wer = compute_asr_scores(references, predictions)

# === Print Results ===
print(f"Character Error Rate (CER): {cer*100:.2f}%")
print(f"Word Error Rate (WER): {wer*100:.2f}%")
