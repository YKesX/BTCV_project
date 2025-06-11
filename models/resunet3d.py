import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet
from monai.networks.layers import Norm

class Net(nn.Module):
    """
    3D ResUNet model for BTCV medical image segmentation
    
    Architecture details:
    - Input channels: 1 (CT image)
    - Output channels: 2 (background + colon cancer)
    - Spatial dimensions: 3 (3D volumes)
    - Feature channels: (16, 32, 64, 128, 256)
    - Using batch normalization
    - Dropout for regularization
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # Initialize the MONAI BasicUNet as a ResUNet alternative - Memory optimized for 8GB GPU
        self.model = BasicUNet(
            spatial_dims=3,           # 3D images
            in_channels=1,            # CT input (single channel)
            out_channels=2,           # Background + colon cancer mask
            features=(16, 32, 64, 128, 256, 512),  # Memory-optimized feature channels (50% smaller)
            norm=Norm.BATCH,          # Use batch normalization
            dropout=0.2,              # Apply dropout for regularization
            act=("LEAKYRELU", {"inplace": True, "negative_slope": 0.01})  # Activation function
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
    Factory function to create and initialize the 3D ResUNet model
    
    Returns:
        Initialized 3D ResUNet model ready for training
    """
    model = Net()
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
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
    
    # Compare with regular UNet
    from models.unet3d import get_model as get_unet_model
    unet_model = get_unet_model().to(device)
    unet_params = sum(p.numel() for p in unet_model.parameters())
    print(f"\nUNet parameters: {unet_params:,}")
    print(f"ResUNet parameters: {total_params:,}")
    print(f"Parameter difference: {total_params - unet_params:,} ({(total_params - unet_params) / unet_params * 100:.2f}%)")