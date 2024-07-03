import transformers
import datasets
import random
import pandas as pd
from datasets import load_dataset
from evaluate import load
import argparse
import nltk
import numpy as np
from transformers import AutoTokenizer, EncoderDecoderModel, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, AutoModel, AutoConfig
from transformers import set_seed
import os
import torch
import random



metric = load("rouge")

parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input sequence length")
parser.add_argument("--max_target_length", type=int, default=256, help="Maximum target sequence length")
parser.add_argument("--eval_steps", type=int, default=1000, help="Maximum target sequence length")
parser.add_argument("--save_steps", type=int, default=1000, help="Maximum target sequence length")

parser.add_argument("--save_total_limit", type=int, default=1, help="Total save checkpoints")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--seed", type=int, default=8, help="seed")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--generation_max_length", type=int, default=256, help="Maximum generation sequence length")
parser.add_argument("--train_vietmed_path", type=str, default="train_whole.json", help="Path to the training dataset")
parser.add_argument("--val_vietmed_path", type=str, default="val_whole.json", help="Path to the validation dataset")
parser.add_argument("--model_checkpoint", type=str, default="VietAI/vit5-base", help="Model checkpoint for initialization")
parser.add_argument("--save_path", type=str, default="~/../../scratch/knguyen07/", help="Path to the save")


args = parser.parse_args()

max_input_length = args.max_input_length
max_target_length = args.max_target_length
batch_size = args.batch_size
learning_rate = args.learning_rate
num_train_epochs = args.num_train_epochs
generation_max_length= args.generation_max_length
train_vietmed_path = args.train_vietmed_path
val_vietmed_path = args.val_vietmed_path
model_checkpoint = args.model_checkpoint
save_total_limit= args.save_total_limit
weight_decay = args.weight_decay
seed = args.seed
save_path = args.save_path


print(f"Random seed set as {seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
transformers.set_seed(seed)
print(f"Random seed set as {seed}")

if 'mbart' in model_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, tgt_lang="en_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
elif 'vihealthbert' in model_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     encoder = AutoModel.from_pretrained(model_checkpoint)
    
#     config = AutoConfig.from_pretrained(model_checkpoint)
#     decoder =  AutoModel.from_config(config)
 
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_checkpoint, model_checkpoint)
    model.config.num_beams = 5
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    max_input_length=256


elif 'vipubmedt5' in model_checkpoint:
    model = AutoModelForSeq2SeqLM.from_pretrained("./vipubmedt5-base-flax", from_flax=True)
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')

else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    
if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""
    
def preprocess_function(examples):
    for i in range(len(examples["summary"])):
        print(examples["summary"][i])
    inputs = [prefix + doc for doc in examples["transcript"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    if 'summary' not in examples:
        examples["summary"] = examples[" summary"]
    for i in range(len(examples["summary"])):
        print(examples["summary"][i])
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_vietmed = load_dataset("json", data_files=train_vietmed_path, split='train').shuffle(seed=seed)
val_vietmed = load_dataset("json", data_files=val_vietmed_path, split='train').shuffle(seed=seed)
train_vietmed_tokenized_datasets = train_vietmed.map(preprocess_function, batched=True)
val_vietmed_tokenized_datasets = val_vietmed.map(preprocess_function, batched=True)


model_name = model_checkpoint.split("/")[-2 if 'checkpoint' in model_checkpoint else -1]
print('Saving the model to '+f"~/../../scratch/knguyen07/{save_path}/{model_name}-vietmed-{train_vietmed_path.split('/')[-1].split('.')[0]}")
train_args = Seq2SeqTrainingArguments(
    f"~/../../scratch/knguyen07/{save_path}/{model_name}-vietmed-{train_vietmed_path.split('/')[-1].split('.')[0]}",
    evaluation_strategy = "steps",
    learning_rate=learning_rate,
    seed=seed,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=save_total_limit,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    generation_max_length=generation_max_length,
    report_to='none',
    fp16=True,#'mt5' not in model_checkpoint,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_avg_rouge',
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result['avg_rouge'] = (result['rouge1'] + result['rouge2'] + result['rougeL'])/3

    return {k: round(v, 4) for k, v in result.items()}

"""Then we just need to pass all of this along with our datasets to the `Seq2SeqTrainer`:"""

trainer = Seq2SeqTrainer(
    model,
    train_args,
    train_dataset=train_vietmed_tokenized_datasets,
    eval_dataset=val_vietmed_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.evaluate()
# trainer.save_model()

"""You can now upload the result of the training to the Hub, just execute this instruction:"""
