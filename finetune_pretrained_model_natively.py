from datasets import load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = load_dataset("yelp_review_full")
# print(dataset["train"][100])

# Add a pretrained tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# apply function with tokenizer to dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove text column because the model does not accept raw text as an input
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Rename the label column to labels because the model expects the argument to be named labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format of the dataset to return PyTorch tensors instead of lists
tokenized_datasets.set_format("torch")

# Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Create a DataLoader for your training and test datasets so you can iterate over batches of data
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# Load your model with the number of expected labels
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

# Create an optimizer and learning rate scheduler to fine-tune the model
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Create the default learning rate scheduler from Trainer
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Lastly, specify device to use a GPU if you have access to one. Otherwise, training on
# a CPU may take several hours instead of a couple of minutes

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training Loop

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# This time instead of calculating the metric one time in the end
# we will accumulate all the batches with add_batch and
# calculate the metric at the very end

import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()



