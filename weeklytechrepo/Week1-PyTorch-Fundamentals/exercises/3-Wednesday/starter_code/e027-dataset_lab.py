import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class DefectDataset(Dataset):
    """
    Task 1: Build the Custom Dataset Class
    """
    def __init__(self, image_paths, labels, transform=None):
        # TODO: Store the arguments as instance variables
        pass

    def __len__(self):
        # TODO: Return the total number of samples
        return 0

    def __getitem__(self, idx):
        # 1. Get the path and label at the specified index
        path = None # TODO
        label = None # TODO
        
        # Simulating loading a PIL image from disk (Random colored square)
        # In reality this would be: image = Image.open(path).convert('RGB')
        raw_image = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))
        
        # 2. Apply the transform if one was provided
        # TODO
            
        # 3. Return the transformed image and the label as a tensor (float32)
        # TODO
        return None, None


def build_pipeline():
    """
    Task 2: Configure Transforms and the DataLoader
    """
    print("--- Building Data Pipeline ---")
    
    # Mock data references
    mock_paths = [f"part_{i}.jpg" for i in range(100)] # 100 images
    mock_labels = [np.random.randint(0, 2) for _ in range(100)] # 0 or 1
    
    # 1. Define the TorchVision Transforms
    # TODO: Compose Resize(128, 128), ToTensor(), and Normalize(mean, std)
    data_transform = None
    
    # 2. Instantiate the Custom Dataset
    # TODO
    dataset = None
    print(f"Dataset securely loaded with {len(dataset)} items.") # Should print 100
    
    # 3. Instantiate the DataLoader
    # TODO: Batch size of 16, shuffle enabled
    dataloader = None
    
    print("\n--- Iterating Pipeline ---")
    # 4. Iterate over the DataLoader for one "epoch"
    # TODO: Loop through dataloader enum and print the shape of the images
    # Example Expected Output: "Batch 1 - Images Shape: torch.Size([16, 3, 128, 128])"
    

if __name__ == "__main__":
    build_pipeline()
