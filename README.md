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

### Dataset Handling (`dataset.py`)

The `dataset.py` module is integral to the data management process for a bilingual text translation task, specifically for translating English to Italian using the "Helsinki-NLP/opus_books" dataset(https://huggingface.co/datasets/Helsinki-NLP/opus_books). This module hosts the `BilingualDataset` class, which prepares and structures the bilingual data for training neural translation models.

#### BilingualDataset Class
This class is at the core of the data handling process, designed to facilitate the efficient preparation of data necessary for training sequence-to-sequence models. It manages the loading, preprocessing, and tokenization of bilingual text data, ensuring the data is properly formatted for use by the neural network.

##### Detailed Functionality
- **Initialization:** Sets up the dataset with the provided tokenizer for both source and target languages, along with defining the sequence length.
- **Data Encoding:** Transforms text data into integer token IDs, ensuring all sequences are of a uniform length as determined by the `seq_len` parameter. Includes mechanisms for handling variable-length sentences by padding shorter sequences and truncating longer ones as necessary.
- **Mask Generation:** Generates attention masks for the transformer model during training. The encoder mask distinguishes padded areas to prevent the model from processing them as part of the sequence. The decoder mask ensures that during prediction, the model can only attend to previously seen tokens, preserving the autoregressive property.

##### Methods
- **`__len__`:** Returns the total number of items in the dataset.
- **`__getitem__`:** Retrieves a single item from the dataset. It processes the source and target text by tokenizing and converting them to integer IDs, applying the necessary padding, and creating the respective masks. Outputs include:
  - `encoder_input`: Sequence of token IDs for the encoder: [`[SOS]` + encoder_input_ids + `[EOS]` + `[PAD]`]
  - `decoder_input`: Prepared token ID sequence for the decoder: [`[SOS]` + decoder_input_ids + `[PAD]`]
  - `label`: Target sequence for model output comparison during training: [decoder_input_ids + `[EOS]` + `[PAD]`]
  - `encoder_mask` and `decoder_mask`: Binary masks indicating valid positions within the encoder and decoder inputs respectively.
  - `src_text` and `tgt_text`: Original text data for reference.

#### Utility Functions
- **`causal_mask`:** Creates a triangular (upper or lower) mask to hide future tokens from being accessed prematurely in sequence predictions. For example:
   ```python
   tensor([[[ True, False, False, False],
            [ True,  True, False, False],
            [ True,  True,  True, False],
            [ True,  True,  True,  True]]])
   ```
- **`get_all_sentences`:** A generator function that iterates over the dataset yielding sentences for tokenizer training.
- **`get_or_build_tokenizer`:** Ensures that a tokenizer is either loaded from disk or trained if not already available, facilitating dynamic preprocessing based on the dataset content.
  - **Initialization**: A new tokenizer is created using a basic word-level model that maps unknown tokens to a special unknown token (`[UNK]`).
  - **Pre-tokenization**: The text is split based on whitespace to simplify the tokenization process.
  - **Trainer Setup**: A trainer is configured with special tokens like `[UNK]`, `[PAD]`, `[SOS]`, and `[EOS]`, and a minimum frequency threshold that tokens must exceed to be included in the vocabulary.
  - **Training**: The tokenizer is trained on sentences from the dataset, which are provided by the `get_all_sentences` function. This function yields one sentence at a time, allowing the tokenizer to learn from actual language usage in the dataset.
  - **Saving**: Once training is complete, the tokenizer is saved to disk at a specified path, ensuring that it can be reloaded and used without needing retraining.

#### Core Function: `get_ds(config)`
- **Purpose:** This function orchestrates the loading, preprocessing, and setup of data loaders for the training and evaluation of the transformer model.
- **Process:**
  1. **Data Acquisition:** Fetches the bilingual dataset from the Hugging Face repository, with options to subsample for experimental purposes.
  2. **Tokenizer Configuration:** Loads or builds tokenizers for both the source and target languages, ensuring text is accurately converted to token IDs.
  3. **Dataset Splitting:** Divides the data into training and validation sets to facilitate model evaluation.
  4. **Dataset Initialization:** Creates instances of `BilingualDataset` for both training and validation datasets with appropriate configurations.
  5. **Data Loader Setup:** Configures data loaders with the correct batch sizes, leveraging parallel loading and shuffling for optimal training performance.
- **Outputs:** Return **`train_dataloader`**, **`val_dataloader`**, **`tokenizer_src`**, **`tokenizer_tgt`**, ready for direct use in training loops.

### Configuration (`config.py`)

The `config.py` module centralizes the configuration settings for the machine translation model, making it easy to manage and adjust parameters across different components of the project. This setup enhances maintainability and scalability of the code by decoupling configuration from implementation.

#### Configuration Details
- **Batch Size:** Defines the number of samples that will be propagated through the network in one pass (`batch_size`).
- **Number of Epochs:** Sets the total number of training cycles through the entire dataset (`num_epochs`).
- **Learning Rate:** Specifies the step size at each iteration while moving toward a minimum of a loss function (`lr`).
- **Sequence Length:** Determines the fixed length of the sequences processed by the model (`seq_len`).
- **Model Dimension:** Specifies the dimensionality of the model's layers (`d_model`).
- **Source Language:** Defines the source language code for translation (`lang_src`).
- **Target Language:** Sets the target language code for translation (`lang_tgt`).
- **Model Folder:** Indicates the directory where trained model weights are stored (`model_folder`).
- **Model Base Name:** Provides the base name for saved model files (`model_basename`).
- **Preload Model Epoch:** Specifies the epoch of the model to load for resuming training or inference (`preload`).
- **Tokenizer File:** Path template for saving or loading tokenizer configurations (`tokenizer_file`).
- **Experiment Name:** Designation for the experiment under which model runs and logs are stored (`experiemnt_name`).
- **Subsample:** A flag to indicate whether to use a subsample of the dataset for experimentation (`subsample`).

#### Helper Functions
- **`get_config`:** Returns a dictionary containing all the configuration settings, facilitating easy access across various modules.
- **`get_weights_file_path`:** Generates the file path for storing or retrieving model weights based on the specified epoch. This function aids in managing different versions of trained models, allowing for flexible loading and storing operations during and after the training process.

#### Usage
The configuration is designed to be imported and used directly in training and evaluation scripts, ensuring that all parts of the project are synchronized in terms of the operational parameters. Changes to any setting require only a single update in this configuration file, simplifying the process of tuning and experimentation.

#### Example
To access the configuration in other modules, you can import and call `get_config()`:
```python
from config import get_config
config = get_config()
```

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


