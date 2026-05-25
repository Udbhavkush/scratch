import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import random
torch.manual_seed(1232)


chars = "1234567890+=."
vocab = sorted(set(chars))
vocab_size = len(vocab)
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

def data_generator():
    a = random.randint(0, 20)
    b = random.randint(0, 20)
    X = f"{a}+{b}="
    Y = f"{a+b}"[::-1]
    return X, Y


def get_batch():
    X, Y = [], []
    batch = map(lambda _: data_generator(), range(5))
    for b in batch:
        X.append(torch.tensor(encode(b[0])))
        Y.append(torch.tensor(encode(b[1])))

    X = pad_sequence(X, padding_value=encode(".")[0], batch_first=True, padding_side='left')
    Y = pad_sequence(Y, padding_value=encode(".")[0], batch_first=True, padding_side='left')

    return X, Y



X, Y = get_batch()

