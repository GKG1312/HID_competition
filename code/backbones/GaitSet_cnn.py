import torch
import torch.nn as nn

class GaitSet(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        """
        GaitSet-inspired model for gait recognition.
        Args:
            num_classes (int): Number of identities.
            hidden_dim (int): Dimension of feature embedding.
        """
        super(GaitSet, self).__init__()
        
        # CNN backbone (simplified ResNet-like)
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
        
        # Global feature extraction
        self.global_fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 1, H, W] (e.g., [8, 30, 1, 128, 128])
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = x.size()
        
        # Process each frame independently
        x = x.view(B * T, C, H, W)  # [B*T, 1, 128, 128]
        features = self.cnn(x)  # [B*T, 128, 16, 16]
        features = features.view(B * T, -1)  # [B*T, 128*16*16]
        features = self.global_fc(features)  # [B*T, hidden_dim]
        
        # Set pooling (max pooling over frames)
        features = features.view(B, T, -1)  # [B, T, hidden_dim]
        features = torch.max(features, dim=1)[0]  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits

if __name__ == "__main__":
    # Test model
    model = GaitSet(num_classes=10)
    x = torch.randn(8, 30, 1, 128, 128)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [8, 10]