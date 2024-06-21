# Medical Spoken Named Entity Recognition
> Please press ⭐ button and/or cite papers if you feel helpful.

This repository contains scripts for automatic speech recognition (ASR) and named entity recognition (NER) using sequence-to-sequence (seq2seq) models and BERT-based models. The provided scripts cover model preparation, training, inference, and evaluation processes, based on the dataset VietMed-NER.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
    - [Seq2Seq Model Training](#seq2seq-model-training)
    - [BERT-Based Model Training](#bert-based-model-training)
  - [Inference](#inference)
    - [Seq2Seq Model Inference](#seq2seq-model-inference)
    - [BERT-Based Model Inference](#bert-based-model-inference)
  - [Evaluation](#evaluation)
- [Contac]

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers
- Datasets
- Tqdm
- Fire
- Loguru

## Setup

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download and prepare the datasets:
    - The scripts expect datasets to be loaded using the `datasets` library. Ensure you have access to the required datasets.

## Usage

### Training

#### Seq2Seq Model Training

1. Prepare the model and tokenizer:
    - Update the `model_name` and other configurations in `seq2seq_models.py` as needed.

2. Run the training script:

    ```bash
    python seq2seq_models.py --train --model_name <model-name>
    ```

#### BERT-Based Model Training

1. Prepare the model and tokenizer:
    - Update the `model_name` and other configurations in `bert_based_models.py` as needed.

2. Run the training script:

    ```bash
    python bert_based_models.py --train --model_name <model-name>
    ```

#### Notes from the PhoBERT developers

- Note that we merged a slow tokenizer for PhoBERT into the main `transformers` branch. The process of merging a fast tokenizer for PhoBERT is in the discussion, as mentioned in this [pull request](https://github.com/huggingface/transformers/pull/17254#issuecomment-1133932067). If users would like to utilize the fast tokenizer, the users might install `transformers` as follows:

```bash
git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
cd transformers
pip3 install -e .
```

- Install tokenizers with pip: `pip3 install tokenizers`

### Inference

#### Seq2Seq Model Inference

1. Prepare the model and tokenizer:
    - Update the `model_path` in `asr_infer_seq2seq.py` with the path to your seq2seq model.

2. Run the inference script:

    ```bash
    python asr_infer_seq2seq.py --model_path <path-to-model>
    ```

#### BERT-Based Model Inference

1. Prepare the model and tokenizer:
    - Update the `model_path` in `asr_infer_bert.py` with the path to your BERT-based model.

2. Run the inference script:

    ```bash
    python asr_infer_bert.py --model_path <path-to-model>
    ```

### Evaluation

1. The evaluation metrics are computed using the `slue.py` and `modified_seqeval.py` scripts.
2. Ensure the scripts are imported correctly in the inference scripts for evaluation.

### Example

1. To train a seq2seq model:

    ```bash
    python seq2seq_models.py --train --model_name facebook/mbart-large-50
    ```

2. To train a BERT-based model:

    ```bash
    python bert_based_models.py --train --model_name bert-base-multilingual-cased
    ```

3. To perform ASR inference using a seq2seq model:

    ```bash
    python asr_infer_seq2seq.py --model_path /path/to/seq2seq_model
    ```

4. To perform ASR inference using a BERT-based model:

    ```bash
    python asr_infer_bert.py --model_path /path/to/bert_model
    ```

## Contact
