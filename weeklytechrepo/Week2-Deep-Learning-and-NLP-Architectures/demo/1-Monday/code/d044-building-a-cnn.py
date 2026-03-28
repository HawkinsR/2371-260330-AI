"""
Demo: Building a Custom ResNet-Style CNN Block
This script demonstrates how to construct a CNN block with convolutions, 
pooling, and a residual (skip) connection. I will also visualize 
the initial feature maps.
"""

import torch
import torch.nn as nn

# ResNet (Residual Network) blocks are the industry standard for modern image models
# They allow for incredibly deep networks by passing original input forward to skip over layers
class ResNetBlock(nn.Module):
    # in_channels: The depth of the incoming image/feature map (e.g., RGB = 3)
    # out_channels: The number of distinct filters applied (e.g., 64)
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        # Primary convolution
        # kernel_size=3 creates a 3x3 pixel scanning window
        # padding=1 ensures the output height/width stays the same as the input (if stride=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        # BatchNorm stabilizes learning by normalizing the activations between layers
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Secondary convolution
        # Stride defaults to 1 here so we don't shrink the image further inside the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection handling if dimensions change
        # A normal shortcut just adds the original x directly to the output.
        self.shortcut = nn.Sequential()
        
        # However, if the stride shrunk the image, or the number of channels changed,
        # we can't directly add x to the output. We must transform x with a 1x1 convolution
        # to match the new dimensions before adding!
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        # Save the original input (x) BEFORE passing it through the convolutions
        identity = self.shortcut(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the skip connection (the original saved input) to the processed output 
        # BEFORE the final ReLU activation. This is what makes it a "Residual" connection!
        out += identity
        out = self.relu(out)
        return out

def visualize_feature_maps():
    print("--- Convolution and Feature Maps Visualization ---")
    
    # 1. Create a dummy image (Batch=1, Channels=3 [RGB], Height=64, Width=64)
    dummy_image = torch.randn(1, 3, 64, 64)
    print(f"Input Image Shape: {dummy_image.shape}")
    
    # 2. Initialize a single Convolutional layer
    # Taking in 3 RGB channels, outputting 4 distinct feature maps (filters)
    # Kernel 3, Padding 1 means the spatial 64x64 dimensions will stay intact
    conv_layer = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
    
    # 3. Pass image through convolution
    feature_maps = conv_layer(dummy_image)
    # Output will be [1, 4, 64, 64] (Batch, OutputChannels, Height, Width)
    print(f"Feature Maps Shape: {feature_maps.shape}")
    
    # 4. Pass through MaxPool
    # MaxPool drops 75% of the data by sliding a 2x2 window and only keeping the highest number
    # This halves the spatial dimensions (64x64 -> 32x32) while retaining the most important features
    pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled_maps = pool_layer(feature_maps)
    
    # Output will be [1, 4, 32, 32]
    print(f"Pooled Feature Maps Shape: {pooled_maps.shape}")
    print("Visualization logic complete.")
    print("-" * 50)

def test_resnet_block():
    print("--- Testing ResNet Block ---")
    
    # Dummy input representing a feature map midway through a network
    x = torch.randn(1, 64, 32, 32)
    print(f"Input Shape: {x.shape}")
    
    # Initialize ResNet block
    # Stride=2 will force the conv layers to half the spatial dimensions (32 -> 16)
    # It also doubles the depth (64 -> 128)
    block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
    
    # Forward pass
    out = block(x)
    # Output should be [1, 128, 16, 16]
    print(f"Output Shape after ResNet Block: {out.shape}")
    print("Notice the spatial dimensions halved and channels doubled.")
    print("-" * 50)

if __name__ == "__main__":
    visualize_feature_maps()
    test_resnet_block()
