import torch
from torch import nn


#In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
#connected feed-forward network, which is applied to each position separately and identically. This
#consists of two linear transformations with a ReLU activation in between.
#FFN(x) = max(0, xW1 + b1)W2 + b2 (2)
#While the linear transformations are the same across different positions, they use different parameters
#from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
#The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.
#以上是论文原文
#进入FFN的时候数据size是[batch, token, 512]
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1, use_residual = True):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual is True:
            result = x + self.ffn(x)
        else:
            result = self.ffn(x)
        return result