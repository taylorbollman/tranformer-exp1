import random
# import pandas as pd
# from IPython.display import display, HTML

import torch
from torch import utils
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import multiprocessing

import accelerate
from accelerate import Accelerator

# import huggingface_hub
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast, T5Tokenizer, AutoTokenizer
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer, default_data_collator, get_scheduler
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


import datasets
from datasets import load_dataset #, load_from_disk
# import evaluate
# from evaluate import load


import tqdm as notebook_tqdm
import os
from dotenv import load_dotenv

from tqdm.auto import tqdm
import math



load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# huggingface_token = os.getenv("HF_TOKEN")

device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device_type)
# print(torch.cuda.device_count())
cpu_cores = multiprocessing.cpu_count()
# print(cpu_cores)
# device = torch.device("cpu")

accelerator = Accelerator()
# device = accelerator.state.device



model_checkpoint = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)



raw_dataset = load_dataset("xsum")



train_size = 1000
test_size = int(0.1 * train_size)
downsampled_dataset = raw_dataset["train"].train_test_split(train_size=train_size, test_size=test_size)




prefix = "summarize: "
max_input_length = 512
max_target_length = 128

# Add prefix to start of each document in dataset. Do NOT use tokenizer() here!
def add_prefix_to_dataset(batch):
    batch["document"] = [prefix + doc for doc in batch["document"]]
    return batch

prefixed_downsampled_dataset = downsampled_dataset.map(add_prefix_to_dataset, batched=True)




def tokenize_dataset(examples):
    # print(examples['document'])  # Add this line
    model_inputs = tokenizer(examples['document'], max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = prefixed_downsampled_dataset.map(tokenize_dataset, batched=True, num_proc=(cpu_cores))
# print(tokenized_datasets)
# print(tokenized_datasets['train'][0])


train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]





batch_size = 64
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='max_length', return_tensors="pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

loss_function = torch.nn.CrossEntropyLoss()



model, optimizer, training_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, training_dataloader, eval_dataloader, lr_scheduler)



num_train_epochs = 2
num_update_steps_per_epoch = len(training_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch


for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(training_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / accelerator.num_processes
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        accelerator.print(f"Epoch: {epoch} | Step: {step} | Loss: {loss}")
