import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from dataloader import get_dataloaders
from backbones.GaitSet_cnn import GaitSet

def train_model(model, dataloaders, num_epochs=50, lr=1e-3, checkpoint_dir='./checkpoints'):
    """
    Train the model.
    Args:
        model: PyTorch model (GaitSet).
        dataloaders: Dict with train/val DataLoaders.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for frames, labels in dataloaders['train']:
            frames, labels = frames.to(device), labels.to(device)
            # Frames are already [B, T, 1, H, W], no need for unsqueeze
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frames, labels in dataloaders['val']:
                frames, labels = frames.to(device), labels.to(device)
                # Frames are already [B, T, 1, H, W]
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * frames.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return model

if __name__ == "__main__":
    # Hyperparameters
    root_dir = r'.\gallery'
    batch_size = 8
    frame_count = 30
    num_epochs = 50
    lr = 1e-3
    
    # Get dataloaders
    dataloaders, label_map = get_dataloaders(root_dir, batch_size, frame_count)
    
    # Initialize model
    model = GaitSet(num_classes=len(label_map))
    
    # Train
    model = train_model(model, dataloaders, num_epochs, lr)