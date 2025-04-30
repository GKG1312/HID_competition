import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaitSetBackbone(nn.Module):
    def __init__(self, hidden_dim=256):
        """
        Backbone for extracting gait embeddings.
        Args:
            hidden_dim (int): Dimension of output embedding.
        """
        super(GaitSetBackbone, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
        )
        
        self.global_fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 1, H, W] (e.g., [8, 30, 1, 128, 128])
        Returns:
            embedding: [B, hidden_dim]
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B * T, -1)
        features = self.global_fc(features)
        features = features.view(B, T, -1)
        embedding = torch.max(features, dim=1)[0]  # [B, hidden_dim]
        return embedding

if __name__ == "__main__":
    # Test models
    backbone = GaitSetBackbone(hidden_dim=256)
    x = torch.randn(8, 30, 1, 128, 128)
    labels = torch.randint(0, 10, (8,))

    # Backbone only
    embedding = backbone(x)
    print(f"Backbone: Embedding shape: {embedding.shape}")