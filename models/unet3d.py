import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

class Net(nn.Module):
    """
    3D U-Net model for BTCV medical image segmentation
    
    Architecture details:
    - Input channels: 1 (CT image)
    - Output channels: 2 (background + colon cancer)
    - Depth: 5 layers
    - Channels: (16, 32, 64, 128, 256)
    - Strides: (2, 2, 2, 2)
    - Using batch normalization
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # Initialize the MONAI UNet with the required parameters
        self.model = UNet(
            spatial_dims=3,           # 3D images
            in_channels=1,            # CT input (single channel)
            out_channels=2,           # Background + colon cancer mask
            channels=(16, 32, 64, 128, 256),  # Feature channels at each layer
            strides=(2, 2, 2, 2),     # Stride for each layer
            num_res_units=3,          # Number of residual units per layer for better gradient flow
            norm=Norm.BATCH,          # Use batch normalization
            dropout=0.2,              # Apply dropout for regularization
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
    Factory function to create and initialize the 3D U-Net model
    
    Returns:
        Initialized 3D U-Net model ready for training
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