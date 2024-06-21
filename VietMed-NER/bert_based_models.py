"""Script to train BERT-based models for NER task and
evaluate the models on the test set."""

import json

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from loguru import logger
import numpy as np
import fire

from slue import get_slue_format, get_ner_scores
from modified_seqeval import classification_report


dataset = load_dataset("yuufong/vietmed_ner_v5")
id2label = dataset["train"].features["tags"].feature._int2str
id2label = {int(k): v for k, v in enumerate(id2label)}
label2id = dataset["train"].features["tags"].feature._str2int


def prepare_model(model_name, device="cuda", is_train=False):
    """Prepare the model for training or evaluation.
    Args:
        model_name (str): The name of the model to be loaded.
        device (str): The device to run the model on.
        is_train (bool): Whether the model is used for training or evaluation.
    Returns:
        model (transformers.modeling_utils.PreTrainedModel): The model to be used.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to be used.
        data_collator (transformers.data.data_collator.DataCollator): The data collator to be used.
        model_name (str): The name of the model to be used."""
    local_files_only = False

    if not is_train:
        model_name = json.load(open(f"{model_name}/config.json", encoding="utf-8"))[
            "_name_or_path"
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        local_files_only = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        local_files_only=local_files_only,
    ).to(device)
    return model, tokenizer, data_collator, model_name


def tokenize_and_align_labels(examples, tokenizer):
    """Tkensize the input and align the labels with the tokens.
    Args:
        examples (dict): The input examples.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to be used.
    Returns:
        tokenized_inputs (dict): The tokenized inputs with aligned labels."""
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def build_compute_metrics(tokenizer):
    """Build the function to compute the metrics for evaluation.
    Args:
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to be used.
    Returns:
        eval_compute_metrics (function): The function to compute the metrics."""

    def eval_compute_metrics(p):
        predictions, labels, inputs = p
        predictions = np.argmax(predictions, axis=2)
        # convert -100 in inputs to 0
        inputs = np.where(inputs == -100, 0, inputs)

        original_words = [
            text.split(" ")
            for text in tokenizer.batch_decode(inputs, skip_special_tokens=True)
        ]

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

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
            for i in range(len(inputs))
        ]
        all_pred_dummy = [
            get_slue_format(original_words[i], true_predictions[i], True)
            for i in range(len(predictions))
        ]

        slue_scores = get_ner_scores(all_gt, all_pred)
        dummy_slue_scores = get_ner_scores(all_gt_dummy, all_pred_dummy)

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


def train_single_model(model_name):
    """Train a single model.
    Args:
        model_name (str): The name of the model to be trained.
    """

    # Add new logger file for the model
    log_id = logger.add(f"logs/{model_name}.log")

    logger.info(f"Loding model {model_name}")
    model, tokenizer, data_collator, model_name = prepare_model(
        model_name, is_train=True
    )

    eval_compute_metrics = build_compute_metrics(tokenizer)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    # prepare training args
    training_args = TrainingArguments(
        output_dir=f"outputs/{model_name}",
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        push_to_hub=True,
        report_to="tensorboard",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        include_inputs_for_metrics=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=eval_compute_metrics,
    )

    logger.info("Start training")
    trainer.train()

    logger.info("Start evaluation")
    result = trainer.predict(tokenized_dataset["test"])
    # logger.info(f"Final result: {result.metrics}")

    logger.info("Finish training, saving the results")
    with open(f"outputs/{model_name}/test_results.json", "w", encoding="utf-8") as f:
        json.dump(result.metrics, f)

    # save the predictions
    # Remove current handler for the logger
    logger.remove(log_id)


def predict_results(model_name):
    """Predict the results of a model.
    Args:
        model_name (str): The name of the model to be evaluated."""
    # Add new logger file for the model
    log_id = logger.add(f"logs/{model_name}.log")

    logger.info(f"Loding model {model_name}")
    model, tokenizer, data_collator, model_name = prepare_model(
        model_name, is_train=False
    )

    eval_compute_metrics = build_compute_metrics(tokenizer)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    # prepare training args
    training_args = TrainingArguments(
        output_dir=f"outputs/{model_name}",
        do_predict=True,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=128,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=eval_compute_metrics,
    )

    logger.info("Start evaluation")
    result = trainer.predict(tokenized_dataset["test"])

    logger.info("Finish training, saving the results")
    with open(f"outputs/{model_name}/test_results.json", "w", encoding="utf-8") as f:
        json.dump(result.metrics, f)

    # save the predictions
    # Remove current handler for the logger
    logger.remove(log_id)


if __name__ == "__main__":
    fire.Fire()
