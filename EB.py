import torch
from torch import nn

class EB(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EB, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        length = x.size(1)
        x = self.token_embedding(x)
        return x