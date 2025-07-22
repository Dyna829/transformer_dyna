import torch
from torch import nn
import math

#回忆一下多头注意力机制
#首先是经过编码的序列要经过三个线性变换变成KQV，这个linear应该就可以
#关键在于shape，传进来的是 [batch, pos, dim]，每个[pos, dim]会进过一个W变成KQV
#经过一个d_model * d_k/d_v的矩阵，
class MHA(nn.Module):
    def __init__(self, d_model, head = 8, use_residual = True, mask = False):
        super(MHA, self).__init__()
        #必须整除
        assert d_model % head == 0, "d_model must be divisible by head"
        self.d_k = d_model // head
        self.d_v = d_model // head
        self.d_model = d_model
        self.head = head
        self.use_residual = use_residual
        self.mask = mask

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)



    def forward(self, x, y):
        Q = self.W_Q(x)
        K = self.W_K(y)
        V = self.W_V(y)

        #这是view的做法
        #multi_head_Q = Q.view(x.size(0), x.size(1), self.head, self.d_k)
        #multi_head_K = K.view(x.size(0), x.size(1), self.head, self.d_k).transpose(2, 3)
        #multi_head_Q = V.view(x.size(0), x.size(1), self.head, self.d_v)

        #更通用的做法是reshape + permute
        multi_head_Q = Q.reshape(x.size(0), x.size(1), self.head, -1).permute(0, 2, 1, 3)
        multi_head_K = K.reshape(y.size(0), y.size(1), self.head, -1).permute(0, 2, 1, 3)
        multi_head_V = V.reshape(y.size(0), y.size(1), self.head, -1).permute(0, 2, 1, 3)

        attention_score = torch.matmul(multi_head_Q, multi_head_K.transpose(2, 3)) / (self.d_k ** 0.5)
        if self.mask is True:
            mask_temp = torch.triu(torch.ones(attention_score.size()), diagonal=1)
            attention_score = attention_score.masked_fill_(mask_temp == 1, float('-inf'))
        Attention = torch.matmul(torch.softmax(attention_score, dim = 3)
                                 , multi_head_V).permute(0, 2, 1, 3).reshape(x.size(0),x.size(1), -1)
        #此时结果还是batch, pos, dim
        if self.use_residual is True:
            return x + self.W_O(Attention)
        else:
            return self.W_O(Attention)

if __name__ == "__main__":
    inp = torch.rand((10,4,64))
    mha = MHA(64, 8)
    oup = mha(inp)
    print(oup)
    print("end\n")
