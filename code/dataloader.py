import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SilhouetteDataset(Dataset):
    def __init__(self, root_dir, frame_count=30, transform=None, mode='train'):
        """
        Args:
            root_dir (str): Path to dataset (e.g., './data').
            frame_count (int): Number of frames to sample per video.
            transform: PyTorch transforms for augmentation/preprocessing.
            mode (str): 'train', 'val', or 'test' for split.
        """
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.transform = transform
        self.mode = mode
        self.videos = []
        self.labels = []
        
        # Create label map
        self.label_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}
        # print(self.label_map)
        
        # Load all .pkl files
        for dirs, folds, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pkl'):
                    # print(dirs.split('\\'))
                    self.videos.append(os.path.join(dirs, file))
                    self.labels.append(self.label_map[dirs.split('\\')[2]])

        # for folder in os.listdir(root_dir):
        #     folder_path = os.path.join(root_dir, folder)
        #     if os.path.isdir(folder_path):
        #         for pkl_file in os.listdir(folder_path):
        #             if pkl_file.endswith('.pkl'):
        #                 self.videos.append(os.path.join(folder_path, pkl_file))
        #                 self.labels.append(self.label_map[folder])
        
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
            frames = pickle.load(f)  # Shape: (n, 128, 128)
        
        # Ensure frames are float32 and normalized
        frames = frames.astype(np.float32) / 255.0
        
        # Sample frames
        n_frames = frames.shape[0]
        if n_frames >= self.frame_count:
            indices = np.linspace(0, n_frames-1, self.frame_count, dtype=int)
            frames = frames[indices]
        else:
            # Pad with last frame
            pad = np.repeat(frames[-1:], self.frame_count - n_frames, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        
        # Apply transforms to each frame (frames are NumPy arrays)
        if self.transform:
            # Convert each frame to PIL Image for torchvision transforms
            frames = [Image.fromarray(frame) for frame in frames]
            frames = torch.stack([self.transform(frame) for frame in frames])  # [T, 1, H, W]
        else:
            # Convert to tensor if no transform
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)  # [T, 1, H, W]
        
        return frames, label

def get_dataloaders(root_dir, batch_size=8, frame_count=30):
    """
    Create train, val, test DataLoaders.
    Args:
        root_dir (str): Dataset path.
        batch_size (int): Batch size.
        frame_count (int): Frames per video.
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    # Define transforms
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),  # Converts [H, W] to [1, H, W]
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=train_transform, mode='train'
    )
    val_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=test_transform, mode='val'
    )
    test_dataset = SilhouetteDataset(
        root_dir, frame_count=frame_count, transform=test_transform, mode='test'
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    return dataloaders, train_dataset.label_map

if __name__ == "__main__":
    # Example usage
    dataloaders, label_map = get_dataloaders(root_dir=r'.\gallery', batch_size=8, frame_count=30)
    print(f"Number of identities: {len(label_map)}")
    for phase in ['train', 'val', 'test']:
        print(f"{phase} samples: {len(dataloaders[phase].dataset)}")