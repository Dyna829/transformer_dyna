import torch

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"src": self.data[idx][0], "tgt": self.data[idx][1]}