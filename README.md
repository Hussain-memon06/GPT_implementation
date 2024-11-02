# GPT_implementation

This repository contains an implementation of a GPT-style transformer model from scratch in PyTorch, following Andrej Karpathy's tutorial. The model includes key components of the transformer architecture, such as multi-head self-attention, feedforward layers, and positional embeddings. This project is designed as a learning resource to understand the foundations of language modeling with transformers.

## Overview

The model is trained on character-level text data, making it suitable for generating text in the style of the input data. In this case, we're using the "tiny Shakespeare" dataset, but the code can be adapted for other datasets as well. This implementation demonstrates a simplified version of the transformer model architecture used in modern language models like GPT.

## Features

- **Bigram Model** (`bigram.py`): A minimalistic baseline model that uses bigrams for text generation.
- **GPT Model** (`gpt.py`): A transformer-based model with multi-head self-attention, feedforward layers, and positional embeddings.
- **Customizable Training Parameters**: Easily adjust batch size, context length, embedding dimensions, and more.
- **Text Generation**: After training, the model can generate new text by predicting subsequent characters based on a given context.

## File Structure

- `bigram.py`: Implementation of a simple bigram language model for quick experimentation.
- `gpt.py`: Main implementation of the GPT-style transformer model, including training and text generation.
- `input.txt`: Sample text data for training the model (e.g., "tiny Shakespeare" dataset).
- `requirements.txt`: Python package requirements for the project.

  ## Text Generation

After training, the model can generate text based on a given context. At the end of `gpt.py`, the following code generates 500 characters of text:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

Modify `max_new_tokens` to change the length of the generated text.

## Model Architecture

The GPT model is implemented with the following components:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds positional information to token embeddings.
- **Multi-Head Self-Attention**: Allows the model to focus on different parts of the sequence simultaneously.
- **Feedforward Layers**: Adds depth and non-linearity to the model.
- **Transformer Blocks**: Stacks of attention and feedforward layers with residual connections.
- **Output Layer**: Maps the final embeddings to a vocabulary-sized output for next-token prediction.

## Hyperparameters

The model includes customizable hyperparameters, such as:

- `batch_size`: Number of sequences processed in parallel.
- `block_size`: Maximum context length for predictions.
- `n_embd`: Embedding dimension.
- `n_head`: Number of attention heads.
- `n_layer`: Number of transformer blocks.
- `learning_rate`: Learning rate for the optimizer.
- `dropout`: Dropout rate to prevent overfitting.

These parameters can be modified in `gpt.py` to explore different configurations.

## Requirements

- Python 3.x
- PyTorch

## Acknowledgements

This implementation is based on Andrej Karpathy’s tutorial on building GPT from scratch. It serves as an educational project for understanding transformers and language modeling.
