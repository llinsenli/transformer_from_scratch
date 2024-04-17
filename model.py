import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    '''
    Vector size of d_model, convey the word tokens from input_ids to corresponding embedding vectors
    init_param: d_model: the dimension of the model
                vocab_size: the vocabulary size
    forward_input: x //(batch_size, seq_len)
    forward_output: tensor in (batch_size, seq_len, d_model)
    '''
    def __init__(self, d_model: int, vocab_size: int)->None:
        '''
        d_model: the dimension of the model
        vocab_size: the vocabulary size
        nn.Embedding(vocab_size, d_model): Pass an integer index , returns the learnable embedding vector corresponding to those indices.
                                           [3]-->[d_model] or [2, 3, 5]-->[d_model, d_model, d_model]
                                           Return is part of the vocab_matrix(vocab_size, d_model) basing on the corresponding word
        '''
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Define the embedding layer, once defined, pass an integer index will return a vector

    def forward(self, x):
        '''
        For each input_ids element, convey it to a d_model vector
        x: (batch_size, seq_len)
        output: (batch_size, seq_len, d_model)
        '''
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    '''
    Vector size of d_model, convey the information about the position of the token inside the sentence to the size of d_model vector
    Only computed once and reused for every sentence during training and inference
    init_param: d_model: the dimension of the model
                seq_len: the sequence length of the input word token for each sentence
                dropout: the dropout rate for the output
    forward_input: x //(batch_size, seq_len, d_model)
    forward_output: InputEmbeddings tensor add the PositionalEncoding //(batch_size, seq_len, d_model)
    '''
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: # -> None indicates that this method doesn't return any value.
        '''
        d_model: the dimension of the model
        seq_len: the sequence length of the input word token for each sentence
        dropout: the dropout rate for the output
        '''
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len)-->(seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)) # (d_model/2)
        # Apply the sin/cos to even/odd positions
        pe[:,0::2] = torch.sin(position * div_term) # 0, 2, 4, 6, .... even position (seq_len, d_model/2)
        pe[:,1::2] = torch.cos(position * div_term) # 1, 3, 5, 7, .... odd position (seq_len, d_model/2)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model), add the batch dimension

        self.register_buffer('pe', pe) # A tensor you want to keep inside the module, not as a learned parameter but saved when save the model

    def forward(self, x):
        '''
        Add the position encoding to everyword inside the sentence which already with input embedding
        x: (batch_size, seq_len, d_model)
        pe: (1, seq_len, d_model)
            If different input x have different seq_len (x.shape[1]),  slicing operation pe[:, :x.shape[1], :]  ensures to use the relevant portion of pe for each sequence in x 
            eg: if x is (4, 6, 10), pe is (1, 10, 10), then this slicing will only use the first 6 rows ie. (1, 6, 10) and add this to x
        output: (batch_size, seq_len, d_model)
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # This particular tensor is not to learn
        return self.dropout(x)
    
class EmbeddingLayers(nn.Module):

    def __init__(self, input_embedding: InputEmbeddings, position_embedding: PositionalEncoding) -> None:
        super().__init__()
        self.input_embedding = input_embedding
        self.position_embedding = position_embedding

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.position_embedding(x)
        return x
    
class LayerNormalization(nn.Module):
    '''
    Place the z-score scale on the last dimension of the input tensor
    init_param: eps: default 10**-6
    forward_input: x //(batch_size, seq_len, d_model) 
    forward_output: a same size tensor as the input x
    '''
    def __init__(self, eps: float = 10**-6) ->None:
        '''
        eps: a small scalar to make sure the denominator not approach 0
        alpha, bias: two learnable scalar to adjust the z-score distribution with some fluctuation, because don't want all value between 0 and 1
        The network will learn to tune these two parameters to introduce fluctuation when necessary
        '''
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Make multiplied parameter learnable in nn.Module
        self.bias = nn.Parameter(torch.zeros(1)) # Make added parameter learnable in nn.Module

    def forward(self, x):
        '''
        Take the z-score scale on the word embedding dim for each word token
        x: (batch_size, seq_len, d_model)
        mean: (batch_size, seq_len, 1)
        std: (batch_size, seq_len, 1)
        output: (batch_size, seq_len, d_model)
        '''
        mean = x.mean(dim = -1, keepdim = True) # take the mean of the last dim, keep the original dim 
        std = x.std(dim = -1, keepdim = True) # take the std of the last dim, keep the original dim 
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias # Element-wise addition with broadcasting

class FeedForwardBlock(nn.Module):
    '''
    A fully connected feed-forward network, which is is applied to each position separately and identically.
    Consists of two linear transformations with ReLU activation between
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float)->None:
        '''
        d_model: the dim of input and output
        d_ff: the dim of inner-layer
        '''
        super().__init__()
        # Define the parameter matrix
        self.linear_1 = nn.Linear(d_model, d_ff,) # W1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        x = self.linear_1(x) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff)
        x = torch.relu(x)
        x = self.dropout(x) 
        x = self.linear_2(x) # (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    '''
    Realize the multi-head attention mechanism
    init_param: d_model: the dimention of the model, using to define the parameter matrix: Wq, Wk, Wv, Wo
                h: number of head
                dropout: the dropout rate
    forward_input: q: query input //(batch_size, seq_len, d_model) 
                   k: key input //(batch_size, seq_len, d_model) 
                   v: value input //(batch_size, seq_len, d_model) 
                   mask: mask input specify by the task (encoder or decoder) //(batch_size, 1, seq_len, seq_len) 
    forward_output: a same size tensor as the input q, k, v //(batch_size, seq_len, d_model) 
    '''
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        '''
        d_model: the dim of input and output
        h: the number of head
        '''
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h # The dimention of each head: head_size
        # Define the parameter matrix
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod #  Changes a method so that it belongs to the class itself rather than to instances of the class. The method cannot modify the class state or instance state, but they can perform operations or calculations relevant to the class, independent of any class or instance variables.
    def attention(query, key, value, mask, dropout: nn.Dropout):
        '''
        query, key, value: 4 dim tensor and the head is in the 2nd dim //(batch_size, h, seq_len, d_k)
        mask: mask the position in attention_scores matrix //(batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        dropout: a nn.Module object with defined dropout rate
        '''
        d_k = query.shape[-1] # head_size
        '''
        Scaled Dot-Product Attention:
        # Scaling the dot product of queries (q) and keys (k) by 1/sqrt(head_size) prevents the values from becoming too large,
        # avoiding an overly peaky softmax distribution. This helps maintain a more uniform distribution across the softmax output,
        # which is crucial for effective gradient propagation during training.
        '''
        #  (batch_size, h, seq_len, d_k) * (batch_size, h, d_k, seq_len)  --> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k) # Scaled attention
        if mask is not None:
            '''
            Padding Mask or Look-ahead Mask
            '''
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch_size, h, seq_len, seq_len) * (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, d_k)
        output = attention_scores @ value 
        return output, attention_scores

    def forward(self, q, k, v, mask):
        '''
        1. Apply the linear layer for the input k, q, v to get the corresponding query, key, value
        2. Transfer each query, key, value from (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        3. Apply the attention mechanism on the (batch_size, h, seq_len, d_k) size data
        4. Transfer back the data from (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) -- > (batch_size, seq_len, d_model)
        5. Apply the last linear layer Wo and return the output
        '''
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) -- > (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # know the first dim is batch_size and the last dim is d_model, then use -1 for the second dim
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    '''
    Realize the skip connection
    init_param: dropout: the dropout rate
    forward_input: x: the input tensor for the sublayer
                   sublayer: an nn.Module object, will apply the x by sublayer(x) 
    forward_output: (x + sublayer(x)) as the skip connection output
    '''
    def __init__(self,dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer: nn.Module):
        '''
        x is the input tensor for sublayer
        sublayer is an nn.Module object, a function when the input is x and the output is sublayer(x), like MultiHeadAttentionBlock() or FeedForwardBlock()
        this is to realize x + sublayer(x)
        '''
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    '''
    The encoder block
    init_param: self_attention_block: a MultiHeadAttentionBlock() object for the self-attention
                feed_forward_block: a FeedForwardBlock() object
                dropout: the dropout rate for ResidualConnection(object)
    forward_input: x: the input for the encoder block
                   src_mask: the mask tensor for encoder
    forward_output: the output of encoder block 
    '''
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        '''
        Encoder Block procedure 
        1. Apply the self-attention on the encoder input
        2. Apply the feed_forward_block
        '''
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask = src_mask)) # using lambda because not only need x but also need mask in the MultiHeadAttentionBlock forward function
        x = self.residual_connections[1](x, self.feed_forward_block) # only need the for the input
        return x
    
class Encoder(nn.Module):
    '''
    The encoder
    init_param: layers: an nn.ModuleList contains a number of EncoderBlock() object
    forward_input: x: the input of the encoder, should be the output of the embedding layer
                   mask: the mask tensor for encoder
    forward_output: the output of the encoder
    '''
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class DecoderBlock(nn.Module):
    '''
    The decoder block
    init_param: self_attention_block: a MultiHeadAttentionBlock() object for the self-attention
                cross_attention_block: a MultiHeadAttentionBlock() object for the cross-attention
                feed_forward_block: a FeedForwardBlock() object
                dropout: the dropout rate for ResidualConnection(object)
    forward_input: x: input of the decoder //(batch_size, seq_len, d_model)
                   encoder_output: output of the encoder //(batch_size, seq_len, d_model)
                   src_mask: mask tensor for encoder
                   tgt_mask: mask tensor for decoder
    forward_output: The output of the decoder block
    '''
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,  feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        '''
        Decoder Block procedure 
        1. Apply the self-attention on the decoder input
        2. Apply the cross-attention on the encoder_output and the output from the self-attention on decoder input
        3. Apply the feed_forward_block
        '''
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(q=x, k=x, v=x,mask=tgt_mask)) # need x and mask for the MultiHeadAttentionBlock object
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(q=x, k=encoder_output, v=encoder_output, mask=src_mask)) # need x, encoder_output, mask for the MultiHeadAttentionBlock object
        x = self.residual_connections[2](x, self.feed_forward_block) # only need x for FeedForwardBlock object
        return x
    
class Decoder(nn.Module):
    '''
    The decoder
    init_param: layers: an nn.ModuleList contains a number of DecoderBlock() object
    forward_input: x: the input of the decoder, should be the output of the embedding layer
                   encoder_output: the output of the encoder
                   src_mask: the mask tensor for encoder
                   tgt_mask: the mask tensor for decoder
    forward_output: the output of the encoder

    '''
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x
    
class ProjectionLayer(nn.Module):
    '''
    Project the embedding vector to the vocabulary dimension for each token
    init_param: d_model: the dimension of the model
                vocab_size: the vocabulary size
    forward_input: x //(batch_size, seq_len, d_model)
    forward_output: tensor with the last dim in softmax scale //(batch_size, seq_len, vocab_size)
    '''
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    '''
    The Transformer itself
    init_param: encoder: an Encoder() object
                decoder: an Decoder() object
                src_embed: InputEmbeddings() object for source language
                tgt_embed: InputEmbeddings() object for targetlanguage
                src_pos: PositionalEncoding() object for source language
                tgt_pos: PositionalEncoding() object for target language
                projection_layer: ProjectionLayer() object 
    method: encode, decode, project
    '''
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        '''
        src: source language data //(batch_size, seq_len)
        src_mask: mask for encoder //(batch_size, seq_len)
        '''
        # Create the word embedding //(batch_size, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        # Get the encoder output //(batch_size, seq_len, d_model)
        output = self.encoder(src, src_mask)
        return output

    def decode(self, encoder_output: torch.Tensor, src_mask, tgt, tgt_mask):
        '''
        encoder_output: the output tensor from the encoder //(batch_size, seq_len, d_model)
        src_mask: mask for encoder //(batch_size, seq_len)
        tgt: target language data //(batch_size, seq_len)
        tgt_mask: mask for decoder //(batch_size, seq_len)
        '''
        # Create the word embedding //(batch_size, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # Get the decoder output //(batch_size, seq_len, d_model) 
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output
    
    def project(self, x):
        '''
        x: the output of the decoder //(batch_size, seq_len, d_model)
        '''
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size) 
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block ,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters, make the training faster, no need to start from random value
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, 
                              vocab_tgt_len, 
                              config["seq_len"], 
                              config["seq_len"], # Use the same seq_len for both language
                              config["d_model"]) # Default is 512, but can modify if necessary
    return model 






    