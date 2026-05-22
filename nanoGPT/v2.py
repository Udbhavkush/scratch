import torch
import torch.nn as nn
from torch.nn import functional as F
import os
torch.manual_seed(1337)

# Defining hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

# reading the data
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# test and train split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # we user register_buffer when we don't want to add a class variable as a parameter of the model
    
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)  # B, T, C
        q = self.key(x)  # B, T, C
        d = k.shape[-1]
        # dot product or basically matrix multiplication
        wei = q @ k.transpose(-2, -1) * d**-0.5  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)
        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # there will be num_heads number of weight matrices here each with a dimension of n_embd*head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
        

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # we are adding an intermediate layer instead of directly taking logits from the embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # so each block or each time component get its own positional embedding
        # self.sa_head = Head(n_embd)
        self.sa_heads = MultiheadAttention(4, n_embd // 4) # basically 4 heads of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size) # adding a linear layer
        
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # T, C. So, for each index, I am getting an embedding that has the information of the position
        x = tok_embd + pos_embd # now x has both the information of identity and position. Although not much useful for bigram but conceptually important
        # x = self.sa_head(x) # self attention head (B, T, C)
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # we cannot call like this. This expects Channel (or C) dimension before. So, we have to reshape the logits
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # get the predictions
        for _ in range(max_new_tokens):
            # crop idx to the last bock_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step (cuz bigram model)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
    
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # batch from sample
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))