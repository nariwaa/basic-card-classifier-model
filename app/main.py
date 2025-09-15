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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"gpu device name: {torch.cuda.get_device_name(0)}\n")
else:
    device = torch.device("cpu")
    print(f"no GPU detected, we'll fallback to CPU 😭\n")

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes

data_dir='/app/dataset/train'
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
    ])

dataset = PlayingCardDataset(data_dir=data_dir, transform=transform)

target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}


image, label = dataset[100]
print(image)
