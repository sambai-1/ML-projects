import torch
import torch.nn as nn

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.first = nn.Embedding(vocabulary_size, 16)
        self.final = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        # embedding goes from B x T x embed_dime to B x embed_dim
        x = self.first(x)
        x = torch.mean(x, 1)
        x = self.final(x)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
