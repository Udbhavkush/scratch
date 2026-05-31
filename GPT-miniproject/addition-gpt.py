"""A mini-project that just takes an input of string of addition of two numbers and gives the output
   The project just deals with two 2-digit numbers (between 0 and 20) and gives their sum probabilistically.
   The main aim of this project was to understand the GPT architecture more and apply it to some problem without
   using AI. Everything here in this file is hand-coded. No help of AI has been taken. The idea to solve this problem
   is from Andrej Karpathy's video where he suggested solving this problem in the video description. The concept and 
   some code has been learnt/taken from his video and Github.
"""
# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import random
torch.manual_seed(1232)


chars = "1234567890+="
vocab = sorted(set(chars))
vocab_size = len(vocab)
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# =============== HYPERPARAMETERS ===============
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 300
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 32
# ================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' ,'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(BATCH_SIZE)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def data_generator():
    a = str(random.randint(0, 20))
    b = str(random.randint(0, 20))
    if len(a) == 1:
        a = '0' + a
    if len(b) == 1:
        b = '0' + b
    s = str(int(a) + int(b))
    if len(s) == 1:
        s = '0' + s
    X = f"{a}+{b}={s[::-1]}"
    maskLen = len(f"{a}+{b}=")
    encoded_sum = encode(str(int(a)+int(b))[::-1])
    if len(encoded_sum) == 1:
        encoded_sum = encode('0') + encoded_sum
    Y = [-1] * (maskLen - 1) + encoded_sum + [-1]
    return X, Y

def get_batch(size):
    X, Y = [], []
    batch = map(lambda _: data_generator(), range(size))
    for b in batch:
        X.append(torch.tensor(encode(b[0])))
        Y.append(torch.tensor(b[1]))
    
    X = pad_sequence(X, padding_value=encode("0")[0], batch_first=True, padding_side='left')
    Y = pad_sequence(Y, padding_value=-1, batch_first=True, padding_side='left')

    return X, Y


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Created a new variable called c_attn (causal attention). As input is N_EMBD, output size would 3 * head_size for K, Q, V
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD) # projection layer that goes back in the pathway
        self.head_size = head_size
        self.num_heads = num_heads
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))
    
    def forward(self, x):
        B, T, C = x.shape
        x = self.c_attn(x) # applied a linear layer to fit the shape we desire as per head_size
        q, k, v = x.split(N_EMBD, dim=2) # since, we want to split on the channel dimension
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD), # projection layer
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformers block: communication (done by self-attention) followed by computation (done by feedforward layer)"""
    def __init__(self, n_head):
        super().__init__()
        head_size = N_EMBD // n_head
        self.sa = CausalSelfAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)  # normalizes at token level
        self.ln2 = nn.LayerNorm(N_EMBD)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # added skip connection here
        x = x + self.ffwd(self.ln2(x)) # added skip connection here
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD) # we are adding an intermediate layer instead of directly taking logits from the embedding table
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD) # so each block or each time component get its own positional embedding
        self.blocks = nn.Sequential(
            Block(n_head=4),
            Block(n_head=4),
            Block(n_head=4),
            nn.LayerNorm(N_EMBD),
        )
        self.lm_head = nn.Linear(N_EMBD, vocab_size) # adding a linear layer
        
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=DEVICE)) # T, C. So, for each index, I am getting an embedding that has the information of the position
        x = tok_embd + pos_embd # now x has both the information of identity and position. Although not much useful for bigram but conceptually important
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1) # we cannot call like this. This expects Channel (or C) dimension before. So, we have to reshape the logits
        
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # get the predictions
        for _ in range(max_new_tokens):
            # crop idx to the last bock_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            # focus only on the last time step (cuz bigram model)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1) # we need definitive answer here (deterministic). So better to use argmax instead of samping 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
model = GPT()
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=LR)

for iter in range(MAX_ITERS):
    
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # batch from sample
    xb, yb = get_batch(BATCH_SIZE)
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()

# %%
Xtest, ytest = get_batch(50)

# %%
# need to write the code to inference the model
output = m.generate(Xtest[:, :-2], 2)

# %%
correct = 0
B, T = output.shape
for i in range(B):
    pre = decode(output[i][:-2].tolist())
    post = decode(output[i][-2:].tolist())[::-1]
    print(pre+post)
    gt = decode(ytest[i][-3:-1].tolist())[::-1]
    if gt == post:
        correct += 1

print(f"Accuracy:{(correct/B)*100}%")
# %%
