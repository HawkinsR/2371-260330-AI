import torch
import torch.nn as nn

class PetClassifierCNN(nn.Module):
    """
    Task 1: Build the CNN Architecture
    """
    def __init__(self):
        super(PetClassifierCNN, self).__init__()
        
        # 1. First Convolutional Block
        # TODO: 3 in_channels, 16 out_channels, 3x3 kernel, padding 1
        self.conv1 = None 
        self.relu1 = nn.ReLU()
        # TODO: MaxPool 2x2 kernel, stride 2
        self.pool1 = None 
        
        # 2. Second Convolutional Block
        # TODO: 16 in_channels, 32 out_channels, 3x3 kernel, padding 1
        self.conv2 = None 
        self.relu2 = nn.ReLU()
        # You can reuse the pool layer or define a new one
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Fully Connected (Linear) Output Layer
        # Input image is 64x64. 
        # After Pool 1 -> 32x32
        # After Pool 2 -> 16x16
        # Flattened size = 32 channels * 16 * 16 spatial area
        flattened_size = 32 * 16 * 16
        
        # TODO: Linear layer from flattened_size to 2 output classes
        self.fc = None

    def forward(self, x):
        # TODO: Route x through conv1, relu1, and pool1
        
        # TODO: Route through conv2, relu2, and pool2
        
        # Flatten the spatial dimensions into a 1D vector for the Linear layer
        # Output of pool2 is [Batch, 32, 16, 16]. We want [Batch, 32*16*16]
        x = x.view(x.size(0), -1) 
        
        # TODO: Route through the final fully connected (fc) layer
        
        return x

def test_forward_pass():
    """
    Task 2: Test the Network Dimensionality
    """
    print("--- Testing CNN Forward Pass ---")
    
    # 1. Instantiate the model
    # TODO
    model = None
    
    # 2. Create a dummy image batch (4 images, 3 channels, 64x64 size)
    dummy_input = torch.randn(4, 3, 64, 64)
    print(f"Input Shape: {dummy_input.shape}")
    
    # 3. Pass through the model
    # TODO
    output = None
    
    # 4. Print the final shape
    if output is not None:
        print(f"Output Shape: {output.shape} (Expected: torch.Size([4, 2]))")
    else:
        print("Output is None — did you complete all the TODOs in the forward pass?")

if __name__ == "__main__":
    test_forward_pass()
