import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """
    3D Convolutional block with batch normalization and LeakyReLU activation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolution
        stride: Stride for convolution
        padding: Padding for convolution
        dropout_p: Dropout probability (0 for no dropout)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
    def forward(self, x):
        # Store residual if channels match (skip residual if they don't)
        # In decoder blocks, input and output channels typically won't match
        residual = x if x.shape[1] == self.conv2.out_channels else None
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add residual connection only if input and output channels match exactly
        if residual is not None and residual.shape == x.shape:
            x += residual
        
        x = self.relu2(x)
        x = self.dropout(x)
        
        return x


class PositionalEncoding3D(nn.Module):
    """
    3D positional encoding for transformer input
    
    Args:
        embed_dim: Embedding dimension
        dropout_p: Dropout probability
        max_len: Maximum sequence length
    """
    def __init__(self, embed_dim, dropout_p=0.1, max_len=100):
        super(PositionalEncoding3D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Initialize positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x has shape [batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding3D(nn.Module):
    """
    3D patch embedding for transformer input
    
    Args:
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        patch_size: Size of each patch
    """
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super(PatchEmbedding3D, self).__init__()
        
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x has shape [batch_size, channels, depth, height, width]
        batch_size, channels, depth, height, width = x.shape
        
        # Project into embedding dimension and flatten patches
        x = self.proj(x)  # Shape: [batch, embed_dim, d', h', w']
        
        # Reshape to [batch, embed_dim, num_patches]
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2)  # Shape: [batch, embed_dim, num_patches]
        
        # Transpose to [batch, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x, (d, h, w)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence processing
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_p: Dropout probability
        feedforward_dim: Dimension of feedforward network
    """
    def __init__(self, embed_dim, num_heads=8, num_layers=6, dropout_p=0.1, feedforward_dim=2048):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_p,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x):
        return self.transformer(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Upsampling scale factor
        dropout_p: Dropout probability
    """
    def __init__(self, in_channels, out_channels, scale_factor=2, dropout_p=0.1):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p=dropout_p)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate skip connection with upsampled feature maps
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv(x)
        return x


class HybridModel(nn.Module):
    """
    Hybrid 3D segmentation model combining CNN and Transformer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        embed_dim: Embedding dimension for transformer
        patch_size: Size of patches for transformer
        num_heads: Number of attention heads in transformer
        num_layers: Number of transformer layers
        dropout_p: Dropout probability
    """
    def __init__(
        self,
        in_channels=1, 
        out_channels=2,
        embed_dim=256,
        patch_size=4,
        num_heads=8,
        num_layers=4,
        dropout_p=0.1
    ):
        super(HybridModel, self).__init__()
        
        # Initial feature extraction and downsampling
        self.encoder_blocks = nn.ModuleList([
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        ])
        
        # Skip connections will store the output of each encoder block
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding3D(128, embed_dim, patch_size=patch_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(embed_dim, dropout_p=dropout_p)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p
        )
        
        # Project transformer output back to convolutional features
        self.proj = nn.Linear(embed_dim, 128)  # Match encoder output channels
        
        # Decoder blocks - properly handle concatenated inputs
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(256, 128, dropout_p=dropout_p),        # (128+128) -> 128
            DecoderBlock(128 + 64, 64, dropout_p=dropout_p),    # (128+64) -> 64  
            DecoderBlock(64 + 32, 32, dropout_p=dropout_p),     # (64+32) -> 32
            DecoderBlock(32 + 16, 16, dropout_p=dropout_p)      # (32+16) -> 16
        ])
        
        # Final output layer
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Encoder path with CNN blocks
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Extract 3D patches and embed
        x, (d, h, w) = self.patch_embedding(x)
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Project back to spatial dimension
        x = self.proj(x)
        
        # Reshape to 3D volume
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, 128, d, h, w)  # Explicit 128 channels
        
        # Decoder path with skip connections and spatial alignment
        for idx, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[-(idx+1)]
            
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = decoder(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class Net(nn.Module):
    """
    Hybrid 3D CNN-Transformer model for BTCV medical image segmentation
    
    Architecture details:
    - Input channels: 1 (CT image)
    - Output channels: 2 (background + colon cancer)
    - CNN encoder: Extract hierarchical features
    - Transformer encoder: Model global contextual information
    - CNN decoder: Generate segmentation output
    """
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = HybridModel(
            in_channels=1,
            out_channels=2,
            embed_dim=256,
            patch_size=2,
            num_heads=8,
            num_layers=4,
            dropout_p=0.1
        )
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape [batch_size, 1, D, H, W]
                where D, H, W are the depth, height, and width of the CT volume
                
        Returns:
            Segmentation prediction of shape [batch_size, 2, D, H, W]
        """
        return self.model(x)


def get_model():
    """
    Factory function to create and initialize the hybrid CNN-Transformer model
    
    Returns:
        Initialized hybrid model ready for training
    """
    model = Net()
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    return model


if __name__ == "__main__":
    # Test the model with a random input tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    # Create a sample input (batch_size=2, channels=1, depth=96, height=96, width=96)
    x = torch.randn(2, 1, 96, 96, 96).to(device)
    
    # Forward pass
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        y = model(x)
        end_time.record()
        
        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # convert to seconds
        else:
            inference_time = 0
    
    # Print model summary
    print("Model Architecture:")
    print(model)
    print("\nInput shape:", x.shape)
    print("Output shape:", y.shape)
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Calculate model size in MB (assuming float32 parameters)
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Compare with other models
    from models.unet3d import get_model as get_unet_model
    from models.resunet3d import get_model as get_resunet_model
    from models.attention_unet import get_model as get_attention_unet_model
    
    unet_model = get_unet_model().to(device)
    resunet_model = get_resunet_model().to(device)
    attention_unet_model = get_attention_unet_model().to(device)
    
    unet_params = sum(p.numel() for p in unet_model.parameters())
    resunet_params = sum(p.numel() for p in resunet_model.parameters())
    attention_unet_params = sum(p.numel() for p in attention_unet_model.parameters())
    
    print(f"\nModel parameter comparison:")
    print(f"UNet parameters: {unet_params:,}")
    print(f"ResUNet parameters: {resunet_params:,}")
    print(f"Attention UNet parameters: {attention_unet_params:,}")
    print(f"Hybrid CNN-Transformer parameters: {total_params:,}")
    
    print(f"\nHybrid vs UNet: {total_params - unet_params:,} more parameters ({(total_params - unet_params) / unet_params * 100:.2f}%)")
    print(f"Hybrid vs ResUNet: {total_params - resunet_params:,} more parameters ({(total_params - resunet_params) / resunet_params * 100:.2f}%)")
    print(f"Hybrid vs Attention UNet: {total_params - attention_unet_params:,} more parameters ({(total_params - attention_unet_params) / attention_unet_params * 100:.2f}%)")