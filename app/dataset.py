import torch
from torch.utils.data import Dataset, DataLoader
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

def get_loaders():
    """Create train, validation, and test dataloaders"""
    train_dataset = PlayingCardDataset(config.TRAIN_DIR, transform=transform)
    val_dataset = PlayingCardDataset(config.VALID_DIR, transform=transform)
    test_dataset = PlayingCardDataset(config.TEST_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    # get the class mapping for later use
    class_to_idx = train_dataset.data.class_to_idx
    target_to_class = {v: k for k, v in class_to_idx.items()}
    
    return train_loader, val_loader, test_loader, target_to_class
