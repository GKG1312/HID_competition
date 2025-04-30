import torch
import torch.nn as nn

class VGG3DBackbone(nn.Module):
    def __init__(self, hidden_dim=512):
        super(VGG3DBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, 64, 8, 112, 112]
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, 128, 4, 56, 56]
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, 256, 2, 28, 28]
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, 512, 1, 14, 14]
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # [B, 512, 1, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, C, T, H, W] = [B, 3, 16, 224, 224]
        # print(f"Input shape: {x.shape}")
        x = self.features(x)
        # print(f"After features: {x.shape}")
        x = self.avgpool(x)
        # print(f"After avgpool: {x.shape}")
        x = x.view(x.size(0), -1)  # [B, 512]
        # print(f"After view: {x.shape}")
        x = self.fc(x)  # [B, hidden_dim]
        # print(f"Output shape: {x.shape}")
        return x

if __name__ == "__main__":
    # Test the backbone
    model = VGG3DBackbone(hidden_dim=512)
    input_tensor = torch.randn(4, 3, 16, 224, 224)
    output = model(input_tensor)
    print(f"Final output shape: {output.shape}")