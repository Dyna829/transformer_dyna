#尝试手撕一个transformer
#需要模块如下
#1. 分词 word2vec
#2. 位置编码 done
#3. MHA 要能够实现mask
#4. FFN done
#5 Layer Norm done 和 Res

from MHA import MHA
from PE import PE
from FFN import FFN
from LayerNorm import LayerNorm
from EB import EB
import torch
from torch import nn
from transformers import AutoTokenizer
import math
import torch.nn.init as init

class encoder_block(nn.Module):
    def __init__(self, d_model, d_ff, head):
        super(encoder_block, self).__init__()
        
        self.mha = MHA(d_model, head)
        self.layer_norm = LayerNorm((d_model,))
        self.ffn = FFN(d_model, d_ff)
        
    
    def forward(self, x):
        x = self.mha(x, x)
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = self.layer_norm(x)
        return x

class encoder(nn.Module):
    def __init__(self, block_num, d_model, d_ff, head):
        super(encoder, self).__init__()

        self.encoder = nn.ModuleList([encoder_block(d_model, d_ff, head) for _ in range(block_num)])
    
    def forward(self, x):
        result = x
        for layer in self.encoder:
            result = layer(result)
        
        return result
    
class decoder_block(nn.Module):
    def __init__(self, d_model, d_ff, head = 8):
        super(decoder_block, self).__init__()
        self.mask_mha = MHA(d_model, head, True, True)
        self.layer_norm = LayerNorm((d_model,))
        self.ffn = FFN(d_model, d_ff)
        self.cross_mha = MHA(d_model, head)

    def forward(self, x, memory):
        x = self.mask_mha(x, x)
        x = self.layer_norm(x)
        x = self.cross_mha(x, memory)
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = self.layer_norm(x)
        return x

class decoder(nn.Module):
    def __init__(self, block_num, d_model, d_ff, head = 8):
        super(decoder, self).__init__()

        self.decoder = nn.ModuleList([decoder_block(d_model, d_ff, head) for _ in range(block_num)])

    def forward(self, x, memory):
        result = x
        for layer in self.decoder:
            result = layer(result, memory)
        
        return result


class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, head, tokenizer):
        super(Transformer, self).__init__()
        #分词和word2vec先跳过
        self.d_model = d_model
        self.d_ff = d_ff
        self.head = head
        self.eb = EB(len(tokenizer), d_model)
        #PE
        self.pe = PE(d_model)
        
        self.encoder = encoder(6, d_model, d_ff, head)
        self.decoder = decoder(6, d_model, d_ff, head)
        self.fc_out = nn.Linear(d_model, len(tokenizer))

    def forward(self, src, tgt):
        #embedding
        src = self.eb(src)
        src += self.pe(src)
        tgt = self.eb(tgt)
        tgt += tgt + self.pe(tgt)

        memory = self.encoder(src)
        tgt = self.decoder(tgt, memory)
        output = self.fc_out(tgt)
        return output
    
    def _initialize_weights(self):
        """初始化整个模型"""
        # 1. 应用全局初始化
        self.apply(self._init_weights)
        
        # 2. 嵌入层缩放
        with torch.no_grad():
            self.embedding.weight *= math.sqrt(self.d_model)
        
        # 3. 输出层特殊处理
        init.xavier_uniform_(self.fc_out.weight, gain=0.02)
        init.constant_(self.fc_out.bias, 0)
    
    def _init_weights(self, module):
        """全局初始化函数"""
        if isinstance(module, nn.Linear):
            if module is self.fc_out:  # 输出层已在外部处理
                return
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0, std=0.02)