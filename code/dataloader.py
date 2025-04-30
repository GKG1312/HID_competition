import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.v2 import Identity

# Backbone configurations
backbone_configs = {
    'gaitstar': {
        'frame_count': 40,
        'input_size': (128, 128),
        'channels': 1,
        'model_type': '2d_cnn',
        'pseudo_rgb': False,
        'for_3d_cnn': False
    },
    'gaitset': {
        'frame_count': 30,
        'input_size': (128, 128),
        'channels': 1,
        'model_type': '2d_cnn',
        'pseudo_rgb': False,
        'for_3d_cnn': False
    },
    'iresnet50': {
        'frame_count': 30,
        'input_size': (128, 128),
        'channels': 1,
        'model_type': '2d_cnn',
        'pseudo_rgb': False,
        'for_3d_cnn': False
    },
    'vgg3d': {
        'frame_count': 16,
        'input_size': (128, 128),
        'channels': 3,
        'model_type': '3d_cnn',
        'pseudo_rgb': True,
        'for_3d_cnn': True
    },
    'gaitformer': {
        'frame_count': 16,
        'input_size': (224, 224),
        'channels': 3,
        'model_type': 'transformer',
        'pseudo_rgb': True,
        'for_3d_cnn': False
    }
}

class ToPseudoRGB:
    """Convert a PIL Image to pseudo-RGB by ensuring 3 channels."""
    def __call__(self, img):
        # print(f"PIL image mode: {img.mode}")
        if img.mode in ('RGB', 'RGBA'):
            return img.convert('RGB')
        return img.convert('L').convert('RGB')  # Force grayscale to RGB

class SilhouetteDataset(Dataset):
    def __init__(self, root_dir, frame_count=30, transform=None, mode='train', for_3d_cnn=False, pseudo_rgb=False):
        """
        Args:
            root_dir (str): Path to dataset (e.g., './gallery').
            frame_count (int): Number of frames to sample per video.
            transform: PyTorch transforms for augmentation/preprocessing.
            mode (str): 'train', 'val', or 'test' for split.
            for_3d_cnn (bool): If True, output [batch_size, channels, frames, height, width].
            pseudo_rgb (bool): If True, transform includes ToPseudoRGB.
        """
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.transform = transform
        self.mode = mode
        self.for_3d_cnn = for_3d_cnn
        self.pseudo_rgb = pseudo_rgb
        self.videos = []
        self.labels = []
        
        # Create label map
        self.label_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}
        
        # Load all .pkl files
        for dirs, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pkl'):
                    self.videos.append(os.path.join(dirs, file))
                    self.labels.append(self.label_map[dirs.split('\\')[2]])
        
        # Split dataset (70% train, 15% val, 15% test)
        total_videos = len(self.videos)
        indices = np.random.permutation(total_videos)
        train_end = int(0.7 * total_videos)
        val_end = int(0.85 * total_videos)
        
        if mode == 'train':
            self.indices = indices[:train_end]
        elif mode == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        video_idx = self.indices[idx]
        pkl_path = self.videos[video_idx]
        label = self.labels[video_idx]
        
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
        
        # print(f"Final frames shape: {frames.shape}, Label: {label}, Path: {pkl_path}")
        return frames, torch.tensor(label, dtype=torch.long), pkl_path

def get_dataloaders(root_dir, backbone_type='gaitstar', batch_size=8):
    """
    Create train, val, test DataLoaders based on backbone configuration.
    Args:
        root_dir (str): Dataset path.
        backbone_type (str): Backbone name (e.g., 'gaitstar', 'gaitset').
        batch_size (int): Batch size.
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}, label_map
    """
    if backbone_type not in backbone_configs:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Supported: {list(backbone_configs.keys())}")
    
    config = backbone_configs[backbone_type]
    frame_count = config['frame_count']
    input_size = config['input_size']
    pseudo_rgb = config['pseudo_rgb']
    for_3d_cnn = config['for_3d_cnn']
    
    # Define transforms
    mean = [0.485, 0.456, 0.406] if pseudo_rgb else [0.5]
    std = [0.229, 0.224, 0.225] if pseudo_rgb else [0.5]
    
    train_transform = transforms.Compose([
        transforms.Resize(input_size) if input_size != (128, 128) else Identity(),
        ToPseudoRGB() if pseudo_rgb else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomHorizontalFlip(p=0.5) if not for_3d_cnn else Identity()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(input_size) if input_size != (128, 128) else Identity(),
        ToPseudoRGB() if pseudo_rgb else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create datasets
    train_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=train_transform, mode='train',
        for_3d_cnn=for_3d_cnn, pseudo_rgb=pseudo_rgb
    )
    val_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=test_transform, mode='val',
        for_3d_cnn=for_3d_cnn, pseudo_rgb=pseudo_rgb
    )
    test_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=test_transform, mode='test',
        for_3d_cnn=for_3d_cnn, pseudo_rgb=pseudo_rgb
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    }
    
    return dataloaders, train_dataset.label_map

if __name__ == "__main__":
    dataloaders, label_map = get_dataloaders(root_dir=r'.\gallery', backbone_type='gaitstar', batch_size=8)
    print(f"Number of identities: {len(label_map)}")
    for phase in ['train', 'val', 'test']:
        print(f"{phase} samples: {len(dataloaders[phase].dataset)}, batches: {len(dataloaders[phase])}")