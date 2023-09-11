import torch
import torch.nn as nn
import torch.nn.functional as F

# v0.1:
# Super simple, hand-written Transformer model. 
# Inspired by the "Building a GPT" notebook by Andrej Karpathy.
# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=False)
    self.query = nn.Linear(d_model, head_size, bias=False)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,C)
    q = self.query(x) # (B,T,C)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
    # apply the "causal mask"
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, n_head, head_size, d_model, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size=head_size,
                                     d_model=d_model,
                                     block_size=block_size,
                                     dropout=dropout) for _ in range(n_head)])
    self.proj = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    # print("MHA output shape", out.shape)
    return out # (B, T, n_embed)

class FeedFoward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.ReLU(),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, d_model, n_head, dropout, block_size):
    # d_model: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = d_model // n_head
    self.sa = MultiHeadAttention(n_head=n_head,
                                 head_size=head_size,
                                 d_model=d_model,
                                 dropout=dropout,
                                 block_size=block_size)
    self.ffwd = FeedFoward(d_model=d_model, dropout=dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    # print("Block output shape", x.shape)
    return x   # (B, T, n_embed)

# super simple model
class SanGuoGPTModel(nn.Module):

    def __init__(self, vocab_size, d_model, n_layer, dropout, block_size, n_head, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model=d_model,
                                            n_head=n_head,
                                            dropout=dropout,
                                            block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.device = device
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = dropout
        self.n_head = n_head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)   # loss is a scalar

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    @torch.no_grad()
    def get_embeddings(self, tokens):
       return self.token_embedding_table(tokens)