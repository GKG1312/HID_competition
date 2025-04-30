import torch
import torch.nn as nn
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

class GaitFormerBackbone(nn.Module):
    def __init__(self, hidden_dim=512, frame_count=16, num_transformer_layers=4):
        super(GaitFormerBackbone, self).__init__()
        self.frame_count = frame_count
        
        # CNN Feature Extractor for frame-level features
        self.conv_blocks = nn.Sequential(
            BasicConvBlock(3, 64, kernel_size=3, stride=2, padding=1),  # [B*T, 64, 112, 112]
            BasicConvBlock(64, 128, kernel_size=3, stride=2, padding=1), # [B*T, 128, 56, 56]
            BasicConvBlock(128, 256, kernel_size=3, stride=2, padding=1), # [B*T, 256, 28, 28]
            nn.AdaptiveAvgPool2d(1)  # [B*T, 256, 1, 1]
        )
        
        # Transformer for temporal modeling
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim=256, num_heads=8, dropout=0.1)
            for _ in range(num_transformer_layers)
        ])
        
        # Class token (learnable parameter)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, frame_count + 1, 256))
        
        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, T, C, H, W] = [B, 16, 3, 224, 224]
        B, T, C, H, W = x.size()
        
        # Reshape for CNN: [B*T, C, H, W]
        x = x.view(B * T, C, H, W)
        
        # Extract frame-level features
        x = self.conv_blocks(x)  # [B*T, 256, 1, 1]
        x = x.view(B, T, 256)  # [B, T, 256]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 256]
        x = torch.cat([cls_token, x], dim=1)  # [B, T+1, 256]
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T+1]
        
        # Transformer: [T+1, B, 256]
        x = rearrange(x, 'b t c -> t b c')
        for layer in self.transformer:
            x = layer(x)
        x = rearrange(x, 't b c -> b t c')  # [B, T+1, 256]
        
        # Use cls_token for final embedding
        x = x[:, 0]  # [B, 256]
        
        # Final projection
        x = self.fc(x)  # [B, hidden_dim]
        return x

if __name__ == "__main__":
    # Test the backbone
    batch_size, frame_count, channels, height, width = 8, 16, 3, 224, 224
    model = GaitFormerBackbone(hidden_dim=512, frame_count=16)
    input_tensor = torch.randn(batch_size, frame_count, channels, height, width)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [8, 512]