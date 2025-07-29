import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformer_dyna import encoder
import math
from torch.nn import init
from LayerNorm import LayerNorm
from torch.optim import AdamW

#写一个vit模块，就是把图片变成序列处理
#一个rgb图片进来是 [batch, channels, height, width]
#接着需要把图像分割成p*p的小块，n = h * w / p^2
#n就是序列长度 现在变成了 [batch, n, p^2 * c],然后通过线性变换变成 d_model就行

class Vi2Seq(nn.Module):
    def __init__(self, height, width, channels, block_size, d_model):
        assert height * width % (block_size ** 2) == 0
        super(Vi2Seq, self).__init__()
        self.height = height
        self.width = width
        self.block_size = block_size
        self.channels = channels * block_size * block_size
        self.seq_length = height * width // (block_size ** 2)
        self.d_model = d_model
        self.linear = nn.Linear(self.channels, d_model)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.channels)
        x = self.linear(x)
        return x


class ViT(nn.Module):
    def __init__(self, height, width, channels, block_size, d_model, d_ff, head, num_classes):
        super(ViT, self).__init__()
        #图像转序列，转完之后维度是d_model
        self.vi2seq = Vi2Seq(height, width, channels, block_size, d_model)
        self.channels = self.vi2seq.channels
        self.seq_length = self.vi2seq.seq_length + 1
        self.num_classes = num_classes
        self.pe = nn.Parameter(torch.zeros(1, self.seq_length, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        #PE，ViT使用的是可学习的
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std= 0.02)
        self.encoder = encoder(6, d_model, d_ff, head)
        self.classifier = nn.Sequential(
            LayerNorm((d_model,)),
            nn.Linear(d_model, self.num_classes)
        )


    def forward(self, x):
        x = self.vi2seq(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pe
        x = self.encoder(x)
        x = x[:, 0, :]
        result = self.classifier(x)
        return result
    
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



if __name__ == "__main__":
    image_path = './image/train'
    val_path = './image/val'
    height = 256
    width = 256
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    data_set = ImageFolder(image_path, transform)
    data_loader = DataLoader(data_set, batch_size = 2, shuffle= True)
    val_data_set = ImageFolder(val_path, transform)
    val_loader = DataLoader(val_data_set, batch_size = 2, shuffle= True)

    model = ViT(height, width, 3, 4, 512, 2048, 8, 2)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )
    epochs = 1
    for epoch in range(0, epochs):
        print("epoch {}:", epoch +1)
        print("training......")
        for images, labels in data_loader:
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
        model.eval()
        total_correct = 0
        total_samples = 0
        print("validating......")
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                # 验证时使用标准交叉熵评估准确率
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()  # 累计正确数
                total_samples += labels.size(0)
        accuracy = total_correct / total_samples
        print(f'Validation Accuracy: {accuracy:.4f}')