import torch
import torch.nn as nn

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first = nn.Linear(28*28, 512)
        self.second = nn.ReLU()
        self.third = nn.Dropout(p=0.2)
        self.fourth = nn.Linear(512, 10)
        self.fifth = nn.Sigmoid()

    
    def forward(self, images):
        torch.manual_seed(0)
        images = self.first(images)
        images = self.second(images)
        images = self.third(images)
        images = self.fourth(images)
        images = self.fifth(images)
        return torch.round(images, decimals=4)
