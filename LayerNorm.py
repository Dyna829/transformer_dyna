# -*- coding: utf-8 -*-
import torch
from torch import nn

#nn里有layernorm 但是还是复现一下巩固一下
#Layer Norm入参首先是张量 size 是 batch， sequence， dimension
#然后是一个epsilon nn里默认是 1e-5，那我也一样呗
#最后是一个可选项，需不需要加上可学习的参数


class LayerNorm(nn.Module):
    def __init__(self, nomalized_shape, epsilon = 1e-5):
        super(LayerNorm, self).__init__()
        self.nomalized_shape = nomalized_shape
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(nomalized_shape))
        self.beta = nn.Parameter(torch.zeros(nomalized_shape))

    def forward(self, x):
        #传入的是一个tuple，算这个tuple的长度然后看最后的维度
        dims = tuple(range(-len(self.nomalized_shape), 0))

        avg = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        nomalized = (x - avg) / std
        result = nomalized * self.gamma + self.beta
        return result

# if __name__=="__main__":
#     x = torch.rand((10, 3, 64))
#     layer = LayerNorm((64,))
#     x = layer(x)
#     print(x)