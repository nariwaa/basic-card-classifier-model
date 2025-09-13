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
        pass
    def __len__(self):
        pass
    def __getitem__(self,idx):
