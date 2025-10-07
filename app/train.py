# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# import from our other files
import config
from model import SimpleCardClassifer
from dataset import get_loaders

def train_model():
    train_loader, val_loader = get_loaders()
    
    model = SimpleCardClassifer(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf') # keep track of the best model

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        
        val_loss = running_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train: {train_loss:.4f}, Valid: {val_loss:.4f}")
        
        # --- THIS IS THE IMPORTANT PART ---
        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"✨ New best model saved to {config.MODEL_PATH}")

if __name__ == '__main__':
    train_model()
