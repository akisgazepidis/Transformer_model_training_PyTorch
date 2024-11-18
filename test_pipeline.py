from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

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
print(encoded_input)