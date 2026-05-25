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
    

class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Created a new variable called c_attn (causal attention). As input is n_embd, output size would 3 * head_size for K, Q, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd) # projection layer that goes back in the pathway
        self.head_size = head_size
        self.num_heads = num_heads
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.shape
        x = self.c_attn(x) # applied a linear layer to fit the shape we desire as per head_size
        q, k, v = x.split(n_embd, dim=2) # since, we want to split on the channel dimension
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, T, nh, head_size) => (B, nh, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        att = q @ k.transpose(-2, -1) * (self.head_size ** (-0.5))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = att @ v # (B, nh, T, T) @ (B, nh, T, head_size) => (B, nh, T, head_size)
        att = att.transpose(1, 2).contiguous().view(B, T, C) # we have to use contiguous() to make the memory in tensors so that view() can work properly
        out = self.proj(att)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity or a basic MLP block"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformers block: communication (done by self-attention) followed by computation (done by feedforward layer)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # normalizes at token level
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # added skip connection here
        x = x + self.ffwd(self.ln2(x)) # added skip connection here
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # we are adding an intermediate layer instead of directly taking logits from the embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # so each block or each time component get its own positional embedding
        self.blocks = nn.Sequential(
            Block(n_embd=n_embd, n_head=4),
            Block(n_embd=n_embd, n_head=4),
            Block(n_embd=n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # adding a linear layer
        
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # T, C. So, for each index, I am getting an embedding that has the information of the position
        x = tok_embd + pos_embd # now x has both the information of identity and position. Although not much useful for bigram but conceptually important
        x = self.blocks(x)
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
    
    
model = GPT()
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

# a small note on skip connections (for my understanding and revision later):
# we have an input x, and we want to learn the function H(x)
# the input x goes through some transformation and we get F(x)
# now we add x to this F(x) we got. The reason we are doing this is first to deal with the problem of vanishing gradients.
# as the network gets deeper, the gradients become smaller, hence vanishing gradient.
# by adding x, we have an addition block that distributes the gradient equally (direct input x and through F(x))
# because of this gradient will not get smaller as it goes through F(x) and we would have some gradient value that comes directly from x.
# Secondly, since, H(x) = F(x) + x, so, F(x) = H(x) - x (This the residual). We have F value by the forward propagation and we need to optimize F(x)
# now, since we just have to optimize the difference, it becomes much easier for the network to do that as it is much more easier to optimize for values ~ 0. 