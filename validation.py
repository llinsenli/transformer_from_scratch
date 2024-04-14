import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import build_transformer, Transformer, get_model
from dataset import causal_mask, get_ds
from config import get_config, get_weights_file_path

import warnings

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    '''
    Autoregressive Decoder: at time step t=0, the decoder_input is (encoder_output, <SOS>), the output is a probability distribution over the vocab_size.
    Search the next token <t0_p> through greedy on this probability distribution. At time t=2, the decoder_input is (encoder_output, [<SOS>, <t0_p>]) and so on. 
    Until the decoder_input reach the max_len or the output token==<EOS>, stop the loop and return the decoder_input.
    '''
    sos_idx = tokenizer_tgt.token_to_id("[SOS]") # The idx of <SOS>
    eos_idx = tokenizer_tgt.token_to_id("[EOS]") # The idx of <EOS>

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask) # The encoder_input and encoder_mask
    # Initial the decoder input with the <SOS>
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # dim = (batch_dim, token_input)
    # Keep to ask decoder to output the next token until reach <EOS> or the max_len 
    while True:
        if decoder_input.size(1) == max_len: # When the input of the decoder reach the max_len
            break

        # Build the mask for the decoder input, we don't want the decoder input to see the future word
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)

        # Get the next token probability
        prob = model.project(out[:, -1]) # (batch_size, vocab_size)
        # Select the token with the max probability (since this is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        # Append the next_word to the decoder_input for the next generatation
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx: # When the next token is the <EOS>
            break
    
    return decoder_input.squeeze(0) # Remove the batch dim using squeeze(0) since the batch dim is 1, the output is a 1 dim tensor
    

def run_validation(model: Transformer, 
                   validation_ds: DataLoader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, num_sample=2):
    model.eval()
    count = 0 # count the number of instance we want to inference

    # source_texts = []
    # expected = []
    # predicted = []

    # Size of the control window (using a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size is 1 for validation"
            # Get the model output token idx 
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device) # dim()==1 tensor

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count ==  num_sample:
                break

def run_inference():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Load the pretrained weights
    model_filename = get_weights_file_path(config, f'05') # Select the stored model
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), global_state=0, num_sample=5)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    run_inference()
    