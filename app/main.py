# # init
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
# print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"gpu device name: {torch.cuda.get_device_name(0)}\n")
else:
    device = torch.device("cpu")
    print(f"no GPU detected, we'll fallback to CPU 😭\n")

# # Pytorch dataset

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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

print(len(dataset))

target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}


image, label = dataset[100]

print(image.shape)

# iterate over dataset
for image, label in dataset:
    break

# ## dataloader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for image, labels in dataloader:
    break
print(image.shape)
print(labels)
print(labels.shape)

# # Pytorch Model

# okay so we're using EfficientNet-B7
class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        # at all but bro said it's ok don't pay attention to it, pretty much it
        # is supposed to remove the last layer (so output layer) from the network so we can
        # have 53 output nodes because the default is 1280 and that's wayyy too large 
        # for our network that is supposed to find cards

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifer(num_classes=53)
print(str(model)[:500])

example_out = model(image)
example_out.shape

# # Training loop

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dir = "./dataset/train/"
valid_dir = "./dataset/valid/"
test_dir = "./dataset/test/"

train_dataset = PlayingCardDataset(train_dir, transform=transform)
val_dataset = PlayingCardDataset(valid_dir, transform=transform)
test_dataset = PlayingCardDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 5
train_losses, val_losses = [], []


model = SimpleCardClassifer(num_classes=53)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
