from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Preprocess Text Data

# Simple example of Tokenizer
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# print(encoded_input)

decoded_input = tokenizer.decode(encoded_input["input_ids"])
# print(decoded_input)


# Example with batch sentences

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences)
# for i in range(len(encoded_input['input_ids'])):
#     print(len(encoded_input['input_ids'][i]))

# Example with batch sentences and Padding

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True)
# for i in range(len(encoded_input['input_ids'])):
#     print(len(encoded_input['input_ids'][i]))

# Example with batch sentences, Padding and Truncation

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
# for i in range(len(encoded_input['input_ids'])):
#     print(len(encoded_input['input_ids'][i]))

# Example with returning the actual tensors 'pt' PyTorch 'tf' Tensorflow

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_input)


# Preprocess Image Data

from transformers import AutoImageProcessor
from datasets import load_dataset
import numpy as np

dataset = load_dataset("food101", split="train[:100]")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True) # There is fast and slow version

# get image
image = dataset[0]["image"]

# Convert the image to a NumPy array
image_array = np.array(image)

# Print the array
# print(image_array)

# Introduce Image Augmentation

# Here we use Compose to chain together a couple of transforms
# - RandomResizedCrop and ColorJitter.
# Note that for resizing, we can get the image size
# requirements from the image_processor.
# For some models, an exact height and width are expected,
# for others only the shortest_edge is defined.

from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])

# The model accepts pixel_values as its input.
# ImageProcessor can take care of normalizing the images,
# and generating appropriate tensors.
# Create a function that combines image augmentation
# and image preprocessing for a batch of images and generates pixel_values


def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images,
                                               # Images already resized. By default, ImageProcessor handles resizing
                                               do_resize=False,
                                               return_tensors="pt"
                                               )["pixel_values"]
    return examples


# apply transforms
dataset.set_transform(transforms)

# print(dataset[0].keys())

# print the image
import matplotlib.pyplot as plt
import torch

im = dataset[0]["pixel_values"]
# Convert to tensor if necessary
if isinstance(im, list):
    img = torch.tensor(im)
# Adjust pixel values to the [0, 1] range
im = (im + 1) / 2

# plt.imshow(im.permute(1, 2, 0))
# plt.axis("off")  # Optional: Hide axes for a cleaner image
# plt.show()  # Ensure the image window stays open
