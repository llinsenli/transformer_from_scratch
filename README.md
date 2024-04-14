# Transformer Model for Machine Translation

This project following the video "https://www.youtube.com/watch?v=ISNdQcPhsts&t=2049s" by Umar Jamil, implements a transformer model from scratch using PyTorch, aimed at performing translation tasks from English to Italian. It leverages the "Helsinki-NLP/opus_books" dataset from Hugging Face for training and evaluation. 

## Overview

The transformer model, introduced in the paper "Attention is All You Need" by Vaswani et al., revolutionized the field of natural language processing by providing a powerful mechanism for sequence-to-sequence tasks. This implementation seeks to recreate the transformer architecture in PyTorch for educational purposes and to demonstrate its capabilities on a translation task.

## Directory Structure

The project consists of several Python files essential to define, train, and evaluate the transformer model:

- `model.py`: Contains the implementation of the transformer model, including the encoder, decoder, and attention mechanisms.
- `dataset.py`: Defines a PyTorch dataset for loading and preprocessing the "Helsinki-NLP/opus_books" dataset for English to Italian translation.
- `config.py`: Stores configuration parameters for the model and training process, making it easy to adjust hyperparameters.
- `train.py`: Orchestrates the training process, including data loading, model instantiation, and the training loop.
- `validation.py`: Implement the validation process, including a greedy_decode method to generate the next token.

### Configuration (`config.py`)

The `config.py` file defines essential configuration settings for the project, such as batch size, learning rate, sequence length, and model dimensions. It also specifies file paths for saving model weights and tokenizer configurations. This setup facilitates easy adjustments and experimentations.

Example usage:
```python
config = get_config()
```

### Dataset Handling (`dataset.py`)

The `dataset.py` file introduces the `BilingualDataset` class, a custom PyTorch `Dataset` for efficiently loading, preprocessing, and tokenizing the bilingual text data. It ensures text pairs are properly encoded with special tokens (`[SOS]`, `[EOS]`, `[PAD]`) and maintains a consistent sequence length across the dataset.

Key functionalities:
- Encoding text pairs to token IDs
- Padding sequences to a fixed length
- Generating causal masks for attention mechanisms

### Training Process (`train.py`)

The `train.py` script manages the model's training cycle. It loads the dataset, initializes the model and optimizer, and executes the training loop. The script also handles splitting the dataset into training and validation sets and saving model checkpoints after each epoch.

Features include:
- Data loading and preprocessing
- Model initialization and training
- Periodic saving of model checkpoints

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or above
- PyTorch
- Hugging Face `datasets`
- (Any other dependencies)

### Installation

1. Clone the repository to your local machine:
   \```bash
   git clone https://github.com/llinsenli/transformer_from_scratch.git
   \```
2. Install the required Python packages:
   \```bash
   pip install torch datasets
   \```

### Training the Model

To train the transformer model on the translation task, navigate to the project directory and run:

\```bash
python train.py
\```

Replace the placeholders in the `config.py` file with your specific model and training settings.

### Evaluation

(Instructions on evaluating the model, possibly including translating sample sentences or calculating performance metrics)

## Contributing

(Instructions for contributing to the project, if applicable)

## License

(Information about the project's license, if applicable)

## Acknowledgments

- The "Attention is All You Need" paper and its authors for introducing the transformer model.
- The Hugging Face team for providing the "Helsinki-NLP/opus_books" dataset.
- (Any other acknowledgments)


