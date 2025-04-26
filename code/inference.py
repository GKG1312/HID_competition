import torch
import pickle
import numpy as np
import os
from models.backbones import GaitSet
import torchvision.transforms as transforms

def load_pkl_frames(pkl_path, frame_count=30):
    """
    Load and preprocess frames from a .pkl file.
    Args:
        pkl_path (str): Path to .pkl file.
        frame_count (int): Number of frames to sample.
    Returns:
        torch.Tensor: [T, 1, H, W]
    """
    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)  # Shape: (n, 128, 128)
    
    frames = frames.astype(np.float32) / 255.0
    n_frames = frames.shape[0]
    
    # Sample frames
    if n_frames >= frame_count:
        indices = np.linspace(0, n_frames-1, frame_count, dtype=int)
        frames = frames[indices]
    else:
        pad = np.repeat(frames[-1:], frame_count - n_frames, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    
    # Convert to tensor
    frames = torch.tensor(frames, dtype=torch.float32)  # [T, H, W]
    transform = transforms.ToTensor()  # [H, W] -> [1, H, W]
    frames = torch.stack([transform(frame) for frame in frames])  # [T, 1, H, W]
    
    return frames

def infer(model, pkl_path, checkpoint_path, label_map, frame_count=30):
    """
    Predict identity for a single .pkl file.
    Args:
        model: PyTorch model (GaitSet).
        pkl_path (str): Path to .pkl file.
        checkpoint_path (str): Path to trained model weights.
        label_map (dict): Mapping of identity names to indices.
        frame_count (int): Number of frames to sample.
    Returns:
        str: Predicted identity name.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Load frames
    frames = load_pkl_frames(pkl_path, frame_count)
    frames = frames.unsqueeze(0).to(device)  # [1, T, 1, H, W]
    
    # Predict
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs, 1)
    
    # Map index to identity name
    reverse_label_map = {v: k for k, v in label_map.items()}
    predicted_identity = reverse_label_map[predicted.item()]
    
    return predicted_identity

if __name__ == "__main__":
    # Example usage
    root_dir = './data'
    pkl_path = './data/Identity_1/video1.pkl'
    checkpoint_path = './checkpoints/best_model.pth'
    frame_count = 30
    
    # Load label map (assuming dataloaders.py is accessible)
    from dataloaders import get_dataloaders
    dataloaders, label_map = get_dataloaders(root_dir, batch_size=8, frame_count=frame_count)
    
    # Initialize model
    model = GaitSet(num_classes=len(label_map))
    
    # Predict
    identity = infer(model, pkl_path, checkpoint_path, label_map, frame_count)
    print(f"Predicted Identity: {identity}")