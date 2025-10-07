import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from model import SimpleCardClassifer
from dataset import get_loaders

def print_system_info():
    """Print system and library versions"""
    print('System Version:', sys.version)
    print('PyTorch version', torch.__version__)
    print('Numpy version', np.__version__)
    print('Pandas version', pd.__version__)
    print()

def train_model():
    print_system_info()
    
    # get dataloaders and class mapping
    train_loader, val_loader, test_loader, target_to_class = get_loaders()
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print()
    
    # initialize model
    model = SimpleCardClassifer(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    print(f"Model architecture:\n{str(model)[:500]}...\n")
    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # tracking
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    # training loop
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} - Training'):
            # Move inputs and labels to the device
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
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
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} - Validation'):
                # Move inputs and labels to the device
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            torch.save(target_to_class, config.CLASS_MAPPING_PATH)
            print(f"✨ New best model saved! (val_loss: {val_loss:.4f})\n")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Class mapping saved to: {config.CLASS_MAPPING_PATH}")

if __name__ == '__main__':
    train_model()
