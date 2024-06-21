"""Infer ASR output with Seq2Seq model and calculate SLUE scores"""

import json

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from loguru import logger
import fire
import numpy as np
from tqdm import tqdm

from slue import get_slue_format, get_ner_scores


# Configuration and Initialization
DEVICE = "cuda"


def load_model_and_tokenizer(model_path, is_train=False):
    """Load the Seq2Seq model and tokenizer
    Args:
        model_path: the path to the model
        is_train: whether the model is for training
    Return:
        model: the model
        tokenizer: the tokenizer
        data_collator: the data collator
        model_name: the model name"""

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model_name = json.load(open(f"{model_path}/config.json"))["_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If the model is mBART, add special tokens
    if "mbart" in model_name.lower():
        tokenizer = config_tokenizer_mbart(tokenizer)
    if is_train:
        model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
    )
    return model, tokenizer, data_collator, model_name


def config_tokenizer_mbart(tokenizer):
    """Config the tokenizer for mBART
    Args:
        tokenizer: the tokenizer
    Return:
        tokenizer: the configured tokenizer"""
    tokenizer.add_special_tokens({"additional_special_tokens": ["<NER>"]})
    tokenizer.src_lang = "vi_VN"
    tokenizer.tgt_lang = "<NER>"
    return tokenizer


# Load dataset and initialize label mappings
dataset = load_dataset("yuufong/vietmed_asr_v3")
id2label_list = (
    dataset["w2v2Viet_Paramshare_LossRem_WERtest_29_0"]
    .features["onehot_gt"]
    .feature._int2str
)
id2label = {int(k): v for k, v in enumerate(id2label_list)}
label2id = (
    dataset["w2v2Viet_Paramshare_LossRem_WERtest_29_0"]
    .features["onehot_gt"]
    .feature._str2int
)
num_labels = len(label2id)
label2id["dum"] = num_labels
id2label[num_labels] = "dum"


def convert_to_seq(words, tags):
    """Convert words and tags to sequence"""
    return " ".join([f"{tag}* {word} {tag}*" for word, tag in zip(words, tags)])


def convert_seq_to_list(words, seq):
    """Convert the sequence to a list of tags and words
    Args:
        words (list): The list of words.
        seq (str): The sequence of tags and words.
    Returns:
        tags (list): The list of tags."""
    predicted_tags = []
    seq_list = seq.split(" ")
    id_in_seq = -1
    # Loop through the words to ensure the number of tags is similar to the number of words
    for word in words:
        # Start from the next index
        id_in_seq += 1
        try:
            # Find the index of the word in the sequence
            id_in_seq = seq_list[id_in_seq:].index(word)
            # Check if the index is not the first or the last index
            if id_in_seq != 0 and id_in_seq != len(seq_list) - 1:
                # Check if the previous and next tags are similar
                if (
                    seq_list[id_in_seq:][id_in_seq - 1]
                    == seq_list[id_in_seq:][id_in_seq + 1]
                ):
                    # valid prediction
                    label_id = int(seq_list[id_in_seq:][id_in_seq - 1].replace("*", ""))
                    label = id2label[label_id]
                    predicted_tags.append(label)
                else:
                    predicted_tags.append("dum")
            else:
                predicted_tags.append("dum")
        except Exception as e:
            logger.warning(f"Error: {e}, assigning dummy tag to the word {word}")
            predicted_tags.append("dum")

    assert len(words) == len(
        predicted_tags
    ), "Length of words and predicted tags must be similar"
    return predicted_tags


def tokenize(example, tokenizer):
    """Tokenize the input and target sequences, adding 'ner:' to the input sequence as the task prefix
    Args:
        example (dict): The input example.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to be used.
    Returns:
        inputs (dict): The tokenized input and target sequences.
    """
    words, tags = example["words"], example["tags"]
    target_seq = [convert_to_seq(word, tag) for word, tag in zip(words, tags)]

    # Check if the tokenizer is mBART
    if "mbart" in tokenizer.name_or_path:
        inputs = tokenizer(
            words,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    else:
        # Add the task prefix to the input sequence
        input_seq = ["ner: " + " ".join(word) for word in words]
        inputs = tokenizer(
            input_seq,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    targets = tokenizer(
        target_seq,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs["labels"] = targets["input_ids"]
    return inputs


def build_eval_compute_metrics(tokenizer):
    """Build the evaluation function to compute the SLUE scores
    Args:
        tokenizer: the tokenizer
    Return:
        eval_compute_metrics: the evaluation function to compute the SLUE scores"""

    def eval_compute_metrics(p):
        predicted_seq, labeled_seq, input_seq = p
        predicted_seq, labeled_seq, input_seq = map(
            lambda seq: np.where(seq == -100, 0, seq),
            [predicted_seq, labeled_seq, input_seq],
        )
        predicted_text = tokenizer.batch_decode(predicted_seq, skip_special_tokens=True)
        labeled_text = tokenizer.batch_decode(labeled_seq, skip_special_tokens=True)
        input_text = tokenizer.batch_decode(input_seq, skip_special_tokens=True)
        original_words = [text.split(" ")[1:] for text in input_text]
        predictions = [
            convert_seq_to_list(word, text)
            for word, text in zip(original_words, predicted_text)
        ]
        labels = [
            convert_seq_to_list(word, text)
            for word, text in zip(original_words, labeled_text)
        ]

        # Pad or truncate the predictions to match the labels
        for i, label in enumerate(labels):
            # Pad with "O" tag if the prediction is shorter than the label
            if len(label) > len(predictions[i]):
                predictions[i] += [label2id["0"]] * (len(label) - len(predictions[i]))
            # Truncate the prediction if it is longer than the label
            elif len(label) < len(predictions[i]):
                predictions[i] = predictions[i][: len(label)]
            else:
                pass

        # Convert the predictions and labels to the list of tags
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # Convert the labels to the list of tags
        true_labels = [
            [id2label[l] for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # Convert the original words, true labels, and true predictions to the SLUE format
        all_gt = [
            get_slue_format(original_words[i], true_labels[i], False)
            for i in range(len(labels))
        ]
        # Convert the original words, true labels, and true predictions to the SLUE format with dummy entities
        all_pred = [
            get_slue_format(original_words[i], true_predictions[i], False)
            for i in range(len(predictions))
        ]
        # Convert the original words, true labels, and true predictions to the SLUE format with dummy entities
        all_gt_dummy = [
            get_slue_format(original_words[i], true_labels[i], True)
            for i in range(len(original_words))
        ]
        # Convert the original words, true labels, and true predictions to the SLUE format with dummy entities
        all_pred_dummy = [
            get_slue_format(original_words[i], true_predictions[i], True)
            for i in range(len(predictions))
        ]
        # Compute the SLUE scores
        slue_scores = get_ner_scores(all_gt, all_pred)
        dummy_slue_scores = get_ner_scores(all_gt_dummy, all_pred_dummy)
        return {"slue_scores": slue_scores, "dummy_slue_scores": dummy_slue_scores}

    return eval_compute_metrics


def prepare_train_args(model_name):
    """Prepare the training arguments for the Seq2Seq model
    Args:
        model_name: the model name
    Return:
        training_args: the training arguments for the Seq2Seq model"""
    return Seq2SeqTrainingArguments(
        output_dir=f"outputs/{model_name}",
        predict_with_generate=True,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        push_to_hub=False,
        report_to="tensorboard",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        include_inputs_for_metrics=True,
        generation_max_length=128,
    )


def predict_results(model_path):
    """Predict the results of the ASR model
    Args:
        model_path: the path to the model
    Return:
        results: the results of the ASR model"""
    model, tokenizer, data_collator, model_name = load_model_and_tokenizer(model_path)
    tokenized_testset = dataset.map(
        tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    compute_metrics = build_eval_compute_metrics(tokenizer)
    training_args = prepare_train_args(model_name)
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    logger.info("Start evaluation")
    for eval_name in [
        "XLSR53Viet",
        "XLSR53Viet_Paramshare_LossRem_WERtest28_8",
        "w2v2Viet_Paramshare_LossRem_WERtest_29_0",
    ]:
        logger.info(f"Evaluating {eval_name} for the model {model_name}")
        result = trainer.predict(tokenized_testset[eval_name])
        with open(f"outputs/asr_eval/{model_name}_{eval_name}.json", "w") as f:
            json.dump(result.metrics, f)


def predict(texts, tokenizer, model, config, batch_size=64):
    """Predict the tags of the texts
    Args:
        texts (list): The list of texts.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to be used.
        model (transformers.modeling_utils.PreTrainedModel): The model to be used.
        config (dict): The generation configuration.
        batch_size (int): The batch size.
    Returns:
        results (list): The list of predicted tags."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_text = texts[i : i + batch_size]
        sequences = [" ".join(text) for text in batch_text]
        inputs = tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        outputs = model.generate(**inputs, generation_config=config)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tags = [
            convert_seq_to_list(text, seq) for text, seq in zip(batch_text, decoded)
        ]
        results.extend(tags)
    return results


def cal_single_ds_ret(sub_ds, tokenizer, model, config, batch_size=32):
    """Calculate the SLUE scores for a single dataset
    Args:
        sub_ds (list): The list of examples in the dataset.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
        The tokenizer to be used.
        model (transformers.modeling_utils.PreTrainedModel): The model to be used.
        config (dict): The generation configuration.
        batch_size (int): The batch size.
    Returns:
        scores (dict): The SLUE scores for the dataset."""

    hyps = [example["hyp"] for example in sub_ds]
    onehot_preds = predict(hyps, tokenizer, model, config, batch_size=batch_size)
    gts = [example["gt"] for example in sub_ds]
    onehot_gts = [example["onehot_gt"] for example in sub_ds]
    all_gt = [get_slue_format(gt, onehot_gt) for gt, onehot_gt in zip(gts, onehot_gts)]
    all_gt_dummy = [
        get_slue_format(gt, onehot_gt, use_dummy=True)
        for gt, onehot_gt in zip(gts, onehot_gts)
    ]
    all_preds = [
        get_slue_format(hyp, onehot_pred)
        for hyp, onehot_pred in zip(hyps, onehot_preds)
    ]
    all_preds_dummy = [
        get_slue_format(hyp, onehot_pred, use_dummy=True)
        for hyp, onehot_pred in zip(hyps, onehot_preds)
    ]
    scores = get_ner_scores(all_gt, all_preds)
    scores_dummy = get_ner_scores(all_gt_dummy, all_preds_dummy)
    return scores, scores_dummy


if __name__ == "__main__":
    fire.Fire(
        {"predict_results": predict_results, "cal_single_ds_ret": cal_single_ds_ret}
    )
