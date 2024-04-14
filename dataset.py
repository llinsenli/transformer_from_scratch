import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from typing import Any

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

class BilingualDataset(Dataset):

    def __init__(self, ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang # "en"
        self.tgt_lang = tgt_lang # "it"
        self.seq_len = seq_len
        # Store the special token's id 
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype = torch.int64) # torch.tensor([2])
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype = torch.int64) # torch.tensor([3])
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64) # torch.tensor([1])
        # Make sure they are in 1 dim tensor
        assert self.sos_token.dim() == 1
        assert self.eos_token.dim() == 1
        assert self.pad_token.dim() == 1
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> Any:
        # Get the source/target text from the ds object
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang] # sentence in str()from source text
        tgt_text = src_tgt_pair['translation'][self.tgt_lang] # sentence in str()from target text

        # Encoding: Convert text --> token --> input_ids(don't include the special token)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # [input_ids] list() from source text for encoder
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # [input_ids] list() from target text for decoder

        # Add "[PAD]" to make the input reach the seq_len to make the inputs in fixed size
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # include the ["SOS"] token and ["EOS"] token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only include the ["SOS"] token, the ["EOS"] token is in the label
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('The sentence is too long')
        
        # Create the encoder input, decoder input, and label in tensor 
        # Source Text: <SOS> + encoder_input_ids + <EOS> + <PAD>...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64) # [1] * 4 --> [1, 1, 1, 1]
            ]
        )
        # Target text: <SOS> + decoder_input_ids + <PAD>...
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        # Label(what we expect as output from the decoder): decoder_input_ids + <EOS> + <PAD>...
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        assert encoder_input.size(0) == self.seq_len and encoder_input.dim() == 1 # (seq_len)
        assert decoder_input.size(0) == self.seq_len and decoder_input.dim() == 1 # (seq_len)
        assert label.size(0) == self.seq_len and label.dim() == 1 # (seq_len)

        # Encoder mask: padding_mask
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        '''
            tensor([[[ 1, 1, 0, 0],
                     [ 1, 1, 0, 0],
                     [ 1, 1, 0, 0],
                     [ 1, 1, 0, 0]]])  
        '''

        # Decoder mask: time_mask & padding_mask
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0) & causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)
        '''
            tensor([[[ 1, 0, 0, 0],                 
                     [ 1, 1, 0, 0],
                     [ 1, 1, 1, 0],
                     [ 1, 1, 1, 1]]]) 
            &
            tensor([[[ 1, 1, 1, 0],                 
                     [ 1, 1, 1, 0],
                     [ 1, 1, 1, 0],
                     [ 1, 1, 1, 0]]]) 
            = 
            tensor([[[ 1, 0, 0, 0],                 
                     [ 1, 1, 0, 0],
                     [ 1, 1, 1, 0],
                     [ 1, 1, 1, 0]]]) 
        '''

        # Return the result as dict() 
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": encoder_mask, # (1, 1, seq_len)
            "decoder_mask": decoder_mask, # (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text, 
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    '''
    Aim to get a mask like this:
    tensor([[[ True, False, False, False],
             [ True,  True, False, False],
             [ True,  True,  True, False],
             [ True,  True,  True,  True]]])
    '''
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) #  .type() with an argument to convert the data type or tensor.to(torch.int32)
    return mask == 0

def get_all_sentences(ds, lang):
    '''
    A generator that yields str, used in tokenizer.train_from_iterator()
    '''
    for item in ds: # each item is a pair of sentences, one in english, another in france
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    if not Path.exists(tokenizer_path): # If the path don't exist, then create one
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # Map the unknown word to '[UNK]'
        tokenizer.pre_tokenizer = Whitespace() # pre_tokenizer means split by ..., here is whitespace
        # Build the trainer to train the tokenizer
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Load the dataset from datasets
def get_ds(config):
    '''
    1. Read the dataset from HuggingFace 
        ds_raw:
            Dataset({
                eatures: ['id', 'translation'],
                num_rows: 32332
            })
    2. Train/load(get_or_build_tokenizer) the two tokenizers for target language and source language
    3. Split the ds into train_ds_raw and val_ds_raw
    4. Create the Dataset object using the defined BilingualDataset class get train_ds, val_ds
    5. Build the train_dataloader and val_dataloader
    6. Return train_dataloader, val_dataloader and two tokenizers
    '''
    ds_raw = load_dataset(path="Helsinki-NLP/opus_books", name=f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') # name ='en-it'
    if config["subsample"]: # Take a subsample for training
        N = 5000  # Number of samples you want to select
        ds_raw = ds_raw.shuffle(seed=42).select(range(N))
        print(f"Experiment on a subsample with size: {N}")
    # Build tokenizers using get_or_build_tokenizer(config, ds, lang)
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) # train a tokenizer for source language
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) # train a tokenizer for target language

    # Split the data: 90% for training, 10% for validation
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # split the ds_raw by [train_ds_size/val_ds_size]

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw , tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Check the max_seq_len in both language
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids 
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids 
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length of the source sentence: {max_len_src}")
    print(f"Max length of the target sentence: {max_len_tgt}")

    # Build the Dataloader object 
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True) # For validation we use the batch_size as 1

    # Return the dataloader and trained tokenizers
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt