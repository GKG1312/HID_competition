import torch
import os
from dataloaders import get_dataloaders
from models.backbones import GaitSet

def test_model(model, test_loader, checkpoint_path, output_dir='./outputs'):
    """
    Test the model.
    Args:
        model: PyTorch model (GaitSet).
        test_loader: DataLoader for test set.
        checkpoint_path (str): Path to trained model weights.
        output_dir (str): Directory to save predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    correct = 0
    total = 0
    predictions = []
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            # Frames are already [B, T, 1, H, W]
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save predictions
    with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    return accuracy

if __name__ == "__main__":
    # Hyperparameters
    root_dir = './data'
    batch_size = 8
    frame_count = 30
    checkpoint_path = './checkpoints/best_model.pth'
    
    # Get test dataloader
    dataloaders, label_map = get_dataloaders(root_dir, batch_size, frame_count)
    test_loader = dataloaders['test']
    
    # Initialize model
    model = GaitSet(num_classes=len(label_map))
    
    # Test
    accuracy = test_model(model, test_loader, checkpoint_path)