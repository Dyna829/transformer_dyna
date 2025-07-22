import torch
from torch import nn
import math

class PE(nn.Module):
    def __init__(self, d_model, max_length = 5000):
        super(PE, self).__init__()
        self.d_model = d_model
        pe = torch.zeros((max_length, d_model))
        pe[0, 1::2] = 1.0
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000) / d_model)
        pe[1, 0::2] = torch.sin(div_term)
        pe[1, 1::2] = torch.cos(div_term)
        for pos in range(2, max_length):
            for i in range(0, d_model//2):
                pe[pos, 2*i] = pe[pos - 1, 2*i] * pe[1, 2*i+1] + pe[pos - 1, 2*i+1] * pe[1, 2*i]
                pe[pos, 2*i+1] = pe[pos - 1, 2*i+1] * pe[1, 2*i+1] - pe[pos - 1, 2*i] * pe[1, 2*i]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length = x.size(1)
        return x + self.pe[:seq_length]
    

#if __name__ == "__main__":
#    inp = torch.rand((10, 3, 64))
#    pe = PE(64)
#   oup = pe(inp)
#    print(oup)
