from datasets import load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = load_dataset("yelp_review_full")
# print(dataset["train"][100])

#Add a pretrained tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# apply function with tokenizer to dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create a smaller dataset so as to reduce time
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load a pretrained classification model and define the number of labels
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

# Define a class with all hyperparameters. I will use the default parameters.
from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer")

# Conduct the evaluation
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the evaluation strategy
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# Create a training object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
