import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for iResNet18."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class BottleneckBlock(nn.Module):
    """Bottleneck residual block for iResNet50 and iResNet100."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class iResNetBackbone(nn.Module):
    """
    iResNet backbone for gait recognition, supporting iResNet18, iResNet50, and iResNet100.
    Args:
        hidden_dim (int): Dimension of output embedding.
        layers (list): Number of blocks in each layer (e.g., [2, 2, 2, 2] for iResNet18).
        block_type (nn.Module): BasicBlock for iResNet18, BottleneckBlock for iResNet50/100.
        in_channels (int): Input channels (1 for grayscale, 3 for pseudo-RGB).
    """
    def __init__(self, hidden_dim=256, layers=[3, 4, 6, 3], block_type=BottleneckBlock, in_channels=1):
        super(iResNetBackbone, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block_type, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2)

        # Global average pooling and fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * block_type.expansion, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] (e.g., [8, 30, 1, 128, 128] or [8, 30, 3, 128, 128])
        Returns:
            embedding: [B, hidden_dim]
        """
        B, T, C, H, W = x.size()
        # Reshape to process all frames independently
        x = x.view(B * T, C, H, W)

        # iResNet forward pass
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(B * T, -1)

        # Fully connected layer
        x = self.fc(x)

        # Reshape and aggregate across temporal dimension
        x = x.view(B, T, -1)
        embedding = torch.max(x, dim=1)[0]  # [B, hidden_dim]
        return embedding

def get_iResNet_backbone(model_type='iresnet50', hidden_dim=256, in_channels=1):
    """
    Factory function to create iResNet backbone.
    Args:
        model_type (str): 'iresnet18', 'iresnet50', or 'iresnet100'.
        hidden_dim (int): Dimension of output embedding.
        in_channels (int): Input channels (1 for grayscale, 3 for pseudo-RGB).
    Returns:
        iResNetBackbone: Configured backbone.
    """
    if model_type == 'iresnet18':
        return iResNetBackbone(
            hidden_dim=hidden_dim,
            layers=[2, 2, 2, 2],
            block_type=BasicBlock,
            in_channels=in_channels
        )
    elif model_type == 'iresnet50':
        return iResNetBackbone(
            hidden_dim=hidden_dim,
            layers=[3, 4, 6, 3],
            block_type=BottleneckBlock,
            in_channels=in_channels
        )
    elif model_type == 'iresnet100':
        return iResNetBackbone(
            hidden_dim=hidden_dim,
            layers=[3, 13, 30, 3],
            block_type=BottleneckBlock,
            in_channels=in_channels
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'iresnet18', 'iresnet50', 'iresnet100'.")

if __name__ == "__main__":
    # Test models
    for model_type in ['iresnet18', 'iresnet50', 'iresnet100']:
        backbone = get_iResNet_backbone(model_type=model_type, hidden_dim=256, in_channels=1)
        x = torch.randn(8, 30, 1, 128, 128)
        embedding = backbone(x)
        print(f"{model_type}: Embedding shape: {embedding.shape}")