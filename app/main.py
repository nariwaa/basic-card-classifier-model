import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform)

    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes

data_dir='/app/dataset/train'
dataset = PlayingCardDataset(data_dir=data_dir)
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
    ])

print(target_to_class)
print(dataset(100))
