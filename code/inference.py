import torch
import os
import csv
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataloader import backbone_configs, SilhouetteDataset, ToPseudoRGB
from train import get_backbone
from classifier import GaitClassifier
from losses import ArcFace
import torchvision.transforms as transforms
from torchvision.transforms.v2 import Identity
import pickle

class InferenceDataset(Dataset):
    def __init__(self, root_dir, frame_count=30, transform=None, for_3d_cnn=False, pseudo_rgb=False):
        """
        Dataset for unlabeled inference .pkl files, yielding sequence tensors.
        Args:
            root_dir (str): Directory containing .pkl files (e.g., './probe_phase1').
            frame_count (int): Number of frames to sample per video.
            transform: PyTorch transforms for preprocessing.
            for_3d_cnn (bool): If True, output [batch_size, channels, frames, height, width] for 3D CNNs.
            pseudo_rgb (bool): If True, transform includes ToPseudoRGB.
        """
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.transform = transform
        self.for_3d_cnn = for_3d_cnn
        self.pseudo_rgb = pseudo_rgb
        self.videos = []
        
        # Recursively find all .pkl files
        for dirs, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pkl'):
                    self.videos.append(os.path.join(dirs, file))
        
        # print(f"Found {len(self.videos)} .pkl files in {root_dir}")
        if not self.videos:
            raise ValueError(f"No .pkl files found in {root_dir}")
        
        # Debug: Print some video paths
        # print(f"Sample video paths: {self.videos[:5] if self.videos else []}")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        pkl_path = self.videos[idx]
        
        # Load .pkl file
        with open(pkl_path, 'rb') as f:
            frames = pickle.load(f)  # Shape: [n, 128, 128]
        
        # Ensure frames are float32 and normalized
        frames = frames.astype(np.float32) / 255.0
        
        # Sample frames
        n_frames = frames.shape[0]
        if n_frames >= self.frame_count:
            indices = np.linspace(0, n_frames-1, self.frame_count, dtype=int)
            frames = frames[indices]
        else:
            pad = np.repeat(frames[-1:], self.frame_count - n_frames, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        
        # Apply transforms to each frame
        if self.transform:
            frames = [Image.fromarray(frame, mode='L') for frame in frames]  # Explicitly set grayscale
            transformed_frames = []
            for frame in frames:
                transformed = self.transform(frame)
                # print(f"Transformed frame shape: {transformed.shape}")
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)  # [T, C, H, W]
        else:
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)  # [T, 1, H, W]
        
        # Permute for 3D CNNs: [T, C, H, W] -> [C, T, H, W]
        if self.for_3d_cnn:
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # print(f"Processed {pkl_path}: Frames shape {frames.shape}")
        return frames, pkl_path

def get_label_map(root_dir):
    """
    Get label map from training dataset without creating dataloaders.
    Args:
        root_dir (str): Path to training dataset (e.g., './gallery').
    Returns:
        dict: Mapping of identity names to indices.
    """
    dataset = SilhouetteDataset(root_dir, frame_count=1, mode='train')  # Minimal config
    return dataset.label_map

def infer_dataset(backbone, classifier, dataset, checkpoint_path, label_map, output_csv='submission.csv'):
    """
    Predict identities for a dataset and save results as videoId, label(identity).
    Args:
        backbone: PyTorch backbone (e.g., GaitSTARBackbone).
        classifier: GaitClassifier with ArcFace loss.
        dataset: InferenceDataset containing .pkl files.
        checkpoint_path (str): Path to trained model checkpoint (.pt).
        label_map (dict): Mapping of identity names to indices.
        output_csv (str): Path to save predictions CSV.
    Returns:
        None (saves results to output_csv).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.eval()
    classifier.eval()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Reverse label map for index-to-identity mapping
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Save predictions to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['videoId', 'label'])
        
        try:
            for frames, video_paths in dataloader:
                frames = frames.to(device)  # [1, T, C, H, W]
                # print(f"Input shape to model: {frames.shape}")
                
                # Predict
                with torch.no_grad():
                    embeddings = backbone(frames)  # [1, hidden_dim]
                    norm_embeddings = torch.nn.functional.normalize(embeddings)
                    norm_weight = torch.nn.functional.normalize(classifier.weight)
                    logits = torch.nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
                    _, predicted = torch.max(logits, 1)
                
                # Get videoId (filename from path)
                video_id = os.path.basename(video_paths[0])
                predicted_identity = reverse_label_map[predicted.item()]
                
                # Write to CSV
                writer.writerow([video_id, int(predicted_identity)])
                print(f"Processed {video_id}: Predicted {predicted_identity}")
        
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, saving predictions and cleaning up...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(0)
        except Exception as e:
            print(f"Error during inference: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(1)
    
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    # Configuration
    inference_dir = r'.\probe_phase1'  # Directory with unlabeled .pkl files
    train_dir = r'.\gallery'  # Training dataset for label_map
    backbone_type = 'iresnet50'  # Options: 'gaitstar', 'gaitset', 'vgg3d', 'gaitformer'
    checkpoint_path = os.path.join(r'.\checkpoints', f'{backbone_type}_best_model_last.pt')  # Dynamic checkpoint path
    output_csv = 'submission.csv'  # Output CSV file
    
    # Get configuration
    if backbone_type not in backbone_configs:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Supported: {list(backbone_configs.keys())}")
    config = backbone_configs[backbone_type]
    frame_count = config['frame_count']
    input_size = config['input_size']
    pseudo_rgb = config['pseudo_rgb']
    for_3d_cnn = config['for_3d_cnn']
    
    # Initialize backbone
    backbone = get_backbone(backbone_type, hidden_dim=512, frame_count=frame_count)
    
    # Get label_map
    try:
        label_map = get_label_map(train_dir)
        num_classes = len(label_map)
        print(f"Number of identities: {num_classes}")
    except Exception as e:
        print(f"Error loading label_map: {e}")
        sys.exit(1)
    
    # Initialize classifier
    loss_fn = ArcFace(s=64.0, margin=0.4)
    classifier = GaitClassifier(margin_loss=loss_fn, embedding_size=512, num_classes=num_classes)
    
    # Create inference dataset
    transform = transforms.Compose([
        transforms.Resize(input_size) if input_size != (128, 128) else Identity(),
        ToPseudoRGB() if pseudo_rgb else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if pseudo_rgb else transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    inference_dataset = InferenceDataset(
        root_dir=inference_dir,
        frame_count=frame_count,
        transform=transform,
        for_3d_cnn=for_3d_cnn,
        pseudo_rgb=pseudo_rgb
    )
    
    # Run inference
    infer_dataset(backbone, classifier, inference_dataset, checkpoint_path, label_map, output_csv)