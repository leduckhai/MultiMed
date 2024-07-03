from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
import torch 
import numpy as np
from tqdm import tqdm
metrics = load_metric('rouge')
import gc
import os


def inference(path):
    prefix = 'summarize: ' if 'mt5' in path else ''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    max_length = 1024 if 'bert' not in path else 256
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["transcript"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=True)
        labels = tokenizer(text_target=examples["summary"], max_length=256, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    dataset = load_dataset("json", data_files="datasets/faq/test/faq_test.json", split='train')
    test_tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    model.to('cuda')


    max_target_length = 256
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(['transcript', 'summary'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_datasets, collate_fn=data_collator, batch_size=32)

    predictions = []
    references = []
    for i, batch in enumerate(tqdm(dataloader)):
      outputs = model.generate(
          input_ids=batch['input_ids'].to('cuda'),
          max_length=max_target_length,
          attention_mask=batch['attention_mask'].to('cuda'),
      )
      with tokenizer.as_target_tokenizer():
        outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

        labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
        actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
      predictions.extend(outputs)
      references.extend(actuals)
      metrics.add_batch(predictions=outputs, references=actuals)

    metrics.compute()

    rouges = [{k: v.mid.fmeasure} for k,v in metrics.compute(predictions=predictions, references=references).items()]
    new_file_path = './r_scores_faq'
    # Write to the file
    try:
        # Attempt to append to the file
        with open(new_file_path, 'a') as file:
            file.write(path.split('/')[-2] + '\n')
            for new_content_str in rouges:
                result = next(iter(new_content_str))
                file.write(f"{result}: {new_content_str[result]}\n")
            file.write('\n')
        action_result = "Content appended to the existing file."
    except FileNotFoundError:
        # File doesn't exist, create it and write the content
        with open(new_file_path, 'w') as file:
            file.write(path)
            file.write(new_content_str)
        action_result = "File did not exist, so it was created with the new content."
    
    del model
    gc.collect()



def get_subdirectories_of_subdirectories(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if 'checkpoint' in full_path and 'df_250' in full_path and 'df_6k' in full_path:
                print(full_path)
                inference(full_path)

# Example usage
directory_path = ''

get_subdirectories_of_subdirectories(directory_path)
