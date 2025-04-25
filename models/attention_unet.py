import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module for 3D medical image segmentation
    
    This implements the attention gate mechanism from the Attention U-Net paper:
    "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., 2018)
    
    Args:
        in_channels: Number of input channels from the skip connection
        gating_channels: Number of channels from the gating signal
        inter_channels: Number of intermediate channels
    """
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        
        # Compression for input features
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        
        # Compression for gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        # Activation function
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x, g):
        # Compress input features and gating signal
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        
        # Align dimensions for addition
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        
        # Apply relu to the sum of compressed inputs
        psi = self.relu(g1 + x1)
        
        # Compute attention map
        psi = self.psi(psi)
        
        # Apply attention map to input feature map
        return x * psi


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers and residual connection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout_p: Dropout probability (0 for no dropout)
    """
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add residual connection if input and output channels match
        if residual.shape[1] == x.shape[1]:
            x += residual
            
        x = self.relu2(x)
        x = self.dropout(x)
        
        return x


class AttentionUNet(nn.Module):
    """
    3D Attention U-Net architecture for medical image segmentation
    
    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        channels: Tuple of feature channels at each level
        strides: Tuple of strides for each downsampling step
        dropout_p: Dropout probability in decoder and bottleneck
    """
    def __init__(self, 
                 spatial_dims=3, 
                 in_channels=1, 
                 out_channels=2, 
                 channels=(16, 32, 64, 128, 256), 
                 strides=(2, 2, 2, 2),
                 dropout_p=0.2):
        super(AttentionUNet, self).__init__()
        
        self.depth = len(channels) - 1
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            in_ch = in_channels if i == 0 else channels[i-1]
            self.encoder.append(ConvBlock(in_ch, channels[i], dropout_p=0))
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels[-2], channels[-1], dropout_p=dropout_p)
        
        # Decoder blocks
        self.decoder = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in range(self.depth):
            # Up-sampling convolutions - from deeper to shallower
            self.up_convs.append(
                nn.ConvTranspose3d(
                    channels[self.depth-i], 
                    channels[self.depth-i-1],
                    kernel_size=strides[self.depth-i-1],
                    stride=strides[self.depth-i-1],
                    bias=True
                )
            )
            
            # Attention gates
            self.attention_gates.append(
                AttentionGate(
                    in_channels=channels[self.depth-i-1],
                    gating_channels=channels[self.depth-i],
                    inter_channels=channels[self.depth-i-1] // 2
                )
            )
            
            # Decoder blocks - combine upsampled and skip features
            self.decoder.append(
                ConvBlock(
                    channels[self.depth-i-1] * 2,  # Concatenated features
                    channels[self.depth-i-1],
                    dropout_p=dropout_p
                )
            )
        
        # Final segmentation layer
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1, bias=True)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i in range(self.depth):
            x = self.encoder[i](x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with attention gates
        for i in range(self.depth):
            # Upsample
            x = self.up_convs[i](x)
            
            # Get skip connection
            skip = skip_connections[self.depth-i-1]
            
            # Apply attention gate
            attended_skip = self.attention_gates[i](skip, x)
            
            # Concatenate with skip connection
            x = torch.cat([x, attended_skip], dim=1)
            
            # Apply decoder conv block
            x = self.decoder[i](x)
        
        # Final 1x1 convolution
        x = self.final_conv(x)
        
        return x


class Net(nn.Module):
    """
    3D Attention U-Net model wrapper for BTCV medical image segmentation
    
    Architecture details:
    - Input channels: 1 (CT image)
    - Output channels: 2 (background + colon cancer)
    - Spatial dimensions: 3 (3D volumes)
    - Feature channels: (16, 32, 64, 128, 256)
    - Strides: (2, 2, 2, 2)
    - Using batch normalization
    - Attention gates in skip connections
    - Dropout for regularization in decoder path
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # Initialize the Attention UNet with the required parameters
        self.model = AttentionUNet(
            spatial_dims=3,           # 3D images
            in_channels=1,            # CT input (single channel)
            out_channels=2,           # Background + colon cancer mask
            channels=(16, 32, 64, 128, 256),  # Feature channels at each layer
            strides=(2, 2, 2, 2),     # Stride for each layer
            dropout_p=0.2,            # Apply dropout for regularization
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
    Factory function to create and initialize the 3D Attention U-Net model
    
    Returns:
        Initialized 3D Attention U-Net model ready for training
    """
    model = Net()
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
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
        y = model(x)
    
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
    
    # Compare with other models
    from models.unet3d import get_model as get_unet_model
    from models.resunet3d import get_model as get_resunet_model
    
    unet_model = get_unet_model().to(device)
    resunet_model = get_resunet_model().to(device)
    
    unet_params = sum(p.numel() for p in unet_model.parameters())
    resunet_params = sum(p.numel() for p in resunet_model.parameters())
    
    print(f"\nModel parameter comparison:")
    print(f"UNet parameters: {unet_params:,}")
    print(f"ResUNet parameters: {resunet_params:,}")
    print(f"Attention UNet parameters: {total_params:,}")
    
    print(f"\nAttention UNet vs UNet: {total_params - unet_params:,} more parameters ({(total_params - unet_params) / unet_params * 100:.2f}%)")
    print(f"Attention UNet vs ResUNet: {total_params - resunet_params:,} more parameters ({(total_params - resunet_params) / resunet_params * 100:.2f}%)")