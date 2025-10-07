# dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import config

# define the transformation
transform = transforms.Compose([
    transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    transforms.ToTensor(),
])

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

# you can also create helper functions to get the dataloaders directly
def get_loaders():
    train_dataset = PlayingCardDataset(config.TRAIN_DIR, transform=transform)
    val_dataset = PlayingCardDataset(config.VALID_DIR, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    return train_loader, val_loader
