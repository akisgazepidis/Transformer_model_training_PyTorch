# Text and Image Preprocessing with Transformers

This guide demonstrates how to preprocess text and image data using Hugging Face Transformers and Datasets libraries.

## Text Preprocessing

### 1. Initialize the Tokenizer
Load the tokenizer for the desired model. For this example, we use `bert-base-cased` to tokenize text.

### 2. Tokenize a Single Sentence
- Tokenize a single sentence into `input_ids` (integer tokens).
- Use the `decode` method to reconstruct the text from the tokens.

### 3. Tokenize Batch Sentences
- Tokenize a list of sentences as a batch.
- Use `padding=True` to ensure all sequences are of equal length.
- Use `truncation=True` to truncate sequences longer than the model’s maximum length.

### 4. Convert Tokens to Tensors
- Set `return_tensors="pt"` to convert the tokenized output into PyTorch tensors for model input.

## Image Preprocessing

### 1. Load the Dataset
- Use the `datasets` library to load an example dataset. In this case, the "food101" dataset is used.
- A small subset is loaded for demonstration (`train[:100]`).

### 2. Process Images
- Use the `AutoImageProcessor` to preprocess images for input into Vision Transformer models.
- The processor handles resizing, normalizing, and tensor conversion.

### 3. Access an Image
- Fetch an image from the dataset using `dataset[0]["image"]`. Depending on the dataset, this may return a PIL image or a NumPy array.

## Example Code
Refer to the script provided to see the implementation of these steps.
