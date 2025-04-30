import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [T, B, C]
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x

class GaitSTARBackbone(nn.Module):
    def __init__(self, hidden_dim=512, frame_count=30, num_transformer_layers=4):
        super(GaitSTARBackbone, self).__init__()
        self.frame_count = frame_count
        
        # CNN Feature Extractor (adapted for [B, T, 1, 128, 128])
        self.conv_blocks = nn.Sequential(
            BasicConvBlock(1, 32, kernel_size=3, stride=1, padding=1),  # [B*T, 32, 128, 128]
            BasicConvBlock(32, 64, kernel_size=3, stride=2, padding=1),  # [B*T, 64, 64, 64]
            ChannelAttention(64),
            BasicConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # [B*T, 128, 32, 32]
            ChannelAttention(128),
            BasicConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # [B*T, 256, 16, 16]
            ChannelAttention(256)
        )
        
        # Spatial Pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # Transformer for Temporal Modeling
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim=256, num_heads=8, dropout=0.1)
            for _ in range(num_transformer_layers)
        ])
        
        # Final Projection
        self.fc = nn.Linear(256, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: [B, T, C, H, W] = [B, 30, 1, 128, 128]
        B, T, C, H, W = x.size()
        
        # Reshape for CNN: [B*T, C, H, W]
        x = x.view(B * T, C, H, W)
        
        # CNN Feature Extraction
        x = self.conv_blocks(x)  # [B*T, 256, 16, 16]
        
        # Spatial Pooling: [B*T, 256, 1, 1]
        x = self.spatial_pool(x).view(B, T, 256)  # [B, T, 256]
        
        # Transformer: [T, B, 256]
        x = rearrange(x, 'b t c -> t b c')
        for layer in self.transformer:
            x = layer(x)
        x = rearrange(x, 't b c -> b t c')  # [B, T, 256]
        
        # Temporal Pooling
        x = x.mean(dim=1)  # [B, 256]
        
        # Final Projection
        x = self.fc(x)  # [B, hidden_dim]
        x = self.bn(x)
        x = self.relu(x)
        
        return x

if __name__ == "__main__":
    # Test the backbone
    batch_size, frame_count, channels, height, width = 8, 30, 1, 128, 128
    model = GaitSTARBackbone(hidden_dim=512, frame_count=30)
    input_tensor = torch.randn(batch_size, frame_count, channels, height, width)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [8, 512]