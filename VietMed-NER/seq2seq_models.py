"""Script to train Seq2Seq models for NER task and
evaluate the models on the test set."""

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
import json

from slue import get_slue_format, get_ner_scores
from modified_seqeval import classification_report

# Load dataset
dataset = load_dataset("yuufong/vietmed_ner_v5")
id2label_list = dataset["train"].features["tags"].feature._int2str
id2label = {int(k): v for k, v in enumerate(id2label_list)}
label2id = dataset["train"].features["tags"].feature._str2int

num_labels = len(label2id)
# Append a dummy label to the label2id and id2label
label2id["dum"] = num_labels
id2label[num_labels] = "dum"


def convert_to_seq(words, tags):
    """Convert words and tags to a sequence of tags and words
    Args:
        words (list): The list of words.
        tags (list): The list of tags.
    Returns:
        seq (str): The sequence of tags and words.

    Example: tags = [1, 2, 3], words = ["I", "am", "fine"]
    Output: "1* I 1* 2* am 2* 3* fine 3*" """

    seq = ""
    for word, tag in zip(words, tags):
        seq += f"{tag}* {word} {tag}* "
    seq = seq.strip()
    return seq


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
    """Build the function to compute the metrics for evaluation
    Args:
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
        The tokenizer to be used.
    Returns:
        eval_compute_metrics (function): The function to compute the metrics."""

    def eval_compute_metrics(p):
        """Compute metrics for the evaluation phase"""
        predicted_seq, labeled_seq, input_seq = p

        # Decode the predicted, labeled, and input sequences
        predicted_text = tokenizer.batch_decode(predicted_seq, skip_special_tokens=True)
        labeled_text = tokenizer.batch_decode(labeled_seq, skip_special_tokens=True)
        input_text = tokenizer.batch_decode(input_seq, skip_special_tokens=True)

        original_words = [
            text.split(" ")[1:] for text in input_text
        ]  # ignore the first token "ner:"
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

        # Convert ids to labels
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Convert the sequences to SLUE format
        all_gt = [
            get_slue_format(original_words[i], true_labels[i], False)
            for i in range(len(labels))
        ]
        all_pred = [
            get_slue_format(original_words[i], true_predictions[i], False)
            for i in range(len(predictions))
        ]
        all_gt_dummy = [
            get_slue_format(original_words[i], true_labels[i], True)
            for i in range(len(original_words))
        ]
        all_pred_dummy = [
            get_slue_format(original_words[i], true_predictions[i], True)
            for i in range(len(predictions))
        ]

        # Compute the SLUE scores
        slue_scores = get_ner_scores(all_gt, all_pred)
        dummy_slue_scores = get_ner_scores(all_gt_dummy, all_pred_dummy)

        # Compute the classification report
        results = classification_report(
            true_predictions, true_labels, digits=4, output_dict=True
        )

        return {
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
            "slue_scores": slue_scores,
            "dummy_slue_scores": dummy_slue_scores,
            "results": results,
        }

    return eval_compute_metrics


def config_tokenizer_mbart(tokenizer):
    """Configure the tokenizer for mBART
    Args:
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
        The tokenizer to be configured.
    Returns:
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
        The configured tokenizer.
    """
    tokenizer.add_special_tokens(
        {"bos_token": "<NER>"}, replace_additional_special_tokens=True
    )
    tokenizer.src_lang = "vi_VN"
    tokenizer.tgt_lang = "<NER>"
    return tokenizer


def prepare_model(model_name, device="cuda", is_train=False):
    """Prepare the model, tokenizer, data collator, and model name
    Args:
        model_name (str): The name of the model to be loaded.
        device (str): The device to be used for training.
        is_train (bool): Whether the model is used for training or evaluation.
    Returns:
        model (transformers.modeling_utils.PreTrainedModel): The model to be used.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
        The tokenizer to be used.
        data_collator (transformers.data.data_collator.DataCollator):
        The data collator to be used.
        model_name (str): The name of the model."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if not is_train:
        model_name = json.load(open(f"{model_name}/config.json"))["_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, src_lang="vi_VN", tgt_lang="vi_VN"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "mbart" in model_name:
        tokenizer = config_tokenizer_mbart(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, tokenizer, data_collator, model_name


def train(
    model_name: str = "facebook/mbart-large-50",
    output_dir: str = "results",
    learning_rate: float = 5e-5,
    epochs: int = 20,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    logging_dir: str = "logs",
):
    """Train the model
    Args:
        model_name (str): The name of the model to be trained.
        output_dir (str): The directory where the model will be saved.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size for the dataloader.
        weight_decay (float): The weight decay for the optimizer.
        logging_dir (str): The directory where the logs will be saved."""

    model, tokenizer, data_collator, model_name = prepare_model(
        model_name, is_train=True
    )

    tokenized_datasets = dataset.map(
        lambda example: tokenize(example, tokenizer), batched=True
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=epochs,
        logging_dir=logging_dir,
        save_strategy="epoch",
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(
    model_dir: str = "results", output_dir: str = "results", split: str = "test"
):
    """Evaluate the model
    Args:
        model_dir (str): The directory where the model is saved.
        output_dir (str): The directory where the evaluation results will be saved.
        split (str): The split to evaluate the model on."""

    model, tokenizer, data_collator, model_name = prepare_model(model_dir)
    tokenized_datasets = dataset.map(
        lambda example: tokenize(example, tokenizer), batched=True
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    eval_result = trainer.evaluate(eval_dataset=tokenized_datasets[split])
    metrics = eval_result.metrics

    logger.info(metrics)

    with open(f"{output_dir}/{model_name}_eval_results.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    fire.Fire({"train": train, "evaluate": evaluate})
