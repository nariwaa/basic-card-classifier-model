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

print(image.shape)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for image, labels in dataloader:
    break
print(image.shape)
print(labels)

# okay so we're using EfficientNet-B7
class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # I don't get this
        # at all but bro said it's ok don't pay attention to it, pretty much it
        # is supposed to remove the last layer (so output layer) from the network so we can
        # have 53 output nodes because the default is 1280 and that's wayyy too large 
        # for our network that is supposed to find cards

        enet_out_size = 1280
        # Classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifer(num_classes=53)
truc = model(image)
print(truc.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dir = "./dataset/train/"
valid_dir = "./dataset/valid/"
test_dir = "./dataset/test/"

train_dataset = PlayingCardDataset(data_dir=train_dir, transform=transform)
valid_dataset = PlayingCardDataset(data_dir=valid_dir, transform=transform)
test_dataset = PlayingCardDataset(data_dir=test_dir, transform=transform)

train_loader = dataloader(train_dataset, batch_size=32, shuffle=True)
valid_loader = dataloader(valid_dataset, batch_size=32, shuffle=False)
test_loader = dataloader(test_dataset, batch_size=32, shuffle=False)

num_epoch = 5
train_loss, val_losses = [], []
model = SimpleCardClassifer(num_classes=53)

for epoch in range(sum_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = models(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.tiem() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    model.eval()

    # validation
    model.eval
    running_loss = 0.0
    with toch.no_grad():
        for image, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
