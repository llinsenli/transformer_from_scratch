import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask, get_ds
from model import build_transformer, get_model
from config import get_config, get_weights_file_path

from tqdm import tqdm
import warnings
from pathlib import Path

def train_model(config):
    '''
    1. Define the device, Get the dataloaders and tokenizers from get_ds(), Get the model
    2. Setup the optimizer and loss_fn
    3. Training loop:
        for epoch in num_epochs:
            for batch in dataloader:
                input_tensors = batch["model_inputs"]
                label = batch["label"]
                model_output = model.forward(input_tensors)
                loss = loss_fn(model_output, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            save_model()
    '''
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Make sure the weight folder is created
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Load our dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    vocab_src_len, vocab_tgt_len = tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size() # Get vocab_size for source/target language
    
    # Build the model
    model = get_model(config, vocab_src_len, vocab_tgt_len).to(device)

    # Tensorboard
    # writer = SummaryWriter(config['experiment_name'])

    # Build the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0 
    # In case the kernal dead, keep on training based on the last training epoch with corresponding model parameter
    if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) # Load the checkpoint from the last training status
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict']) # To keep on training from where we end last time
        optimizer.load_state_dict(state['optimizer_state_dict']) # we need to load the model and the optimizer
        global_step = state['global_step']

    # Loss funciton
    '''
    For each item, the label is (seq_len) tensor [<T1>, <T2>, <T3>, ...<EOS>, <PAD>, <PAD>, ...], 
    while the model project output is (seq_len, vocab_size). We should only focus on the valid tokens(exclude <PAD>), which means 
    the loss_fn should compute the loss for [<T1>, <T2>, <T3>, ...<EOS>] and (:id(<EOS>), vocab_size). 
    Use ignore_index=tokenizer_src.token_to_id('[PAD]') to realize this.
    '''
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Define the training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len), only mask on the last dimension
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)

            # Feed the tensor to the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)
            # Model output: (batch_size, seq_len, tgt_vocab_size) --> (batch_size * seq_len, tgt_vocab_size) # dim=2
            # label: (batch_size, seq_len) --> (batch_size * seq_len) # dim=1,  tensor.view(-1) flattens the entire tensor into a 1-dimensional tensor (a vector)
            loss = loss_fn(proj_output.view(-1, vocab_tgt_len), label.view(-1))
            batch_iterator.set_postfix({f'loss': f"{loss.item(): 6.3f}"})

            # Log the loss
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()

            # Backpropagate the loss and update the weights
            loss.backward() # Backpropagate
            optimizer.step() # Update weights
            optimizer.zero_grad() # Clear gradients

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch, # Store the current epoch
            'model_state_dict': model.state_dict(), # Store the current model parameter
            'optimizer_state_dict': optimizer.state_dict(), # Store the state of the optimizer
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)








