import torch
from torch.utils.data import Dataset, DataLoader  # Essential for handling large batches of data efficiently
from torchvision import transforms  # Contains image processing and augmentation functions
from PIL import Image  # Standard Python Imaging Library used for loading images
import numpy as np

# 1. Simulate a Custom Dataset Class
# All custom datasets in PyTorch MUST inherit from torch.utils.data.Dataset
class FakeImageDataset(Dataset):
    """
    Simulates a dataset where images are stored on disk and loaded dynamically.
    Inherits from torch.utils.data.Dataset.
    """
    # The __init__ method is run once when the dataset object is created.
    # It usually handles reading file paths or CSV labels into memory.
    def __init__(self, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
        # We pretend these are a list of file paths we read from a CSV or directory
        self.image_paths = [f"image_{i}.jpg" for i in range(num_samples)]
        
        # Determine a fake binary label (0 or 1) for each image
        self.labels = [np.random.randint(0, 2) for _ in range(num_samples)] 

    # The __len__ method is mandatory! 
    # It tells the DataLoader how many total items exist in the entire dataset.
    def __len__(self):
        return self.num_samples

    # The __getitem__ method is mandatory!
    # It dictates exactly how to fetch and prepare ONE specific item (at index 'idx')
    def __getitem__(self, idx):
        # 1. Get the path and label for this specific item
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 2. Load the actual data
        # Simulate opening a real image with PIL (Python Imaging Library)
        # We generate a random noise color image of shape [Height, Width, Channels]
        fake_image = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        
        print(f"[Dataset] Loading {img_path}")
        
        # 3. Apply transformations
        # Apply transforms if they were provided during initialization
        if self.transform:
            fake_image = self.transform(fake_image)
            
        # 4. Return a tuple of (data_tensor, label_tensor)
        return fake_image, torch.tensor(label, dtype=torch.float32)

def run_dataloader_demo():
    print("--- 1. Define Transforms ---")
    # transforms.Compose chains multiple data augmentation steps together into one pipeline
    custom_transform = transforms.Compose([
        # Change image size to 32x32 pixels
        transforms.Resize((32, 32)), 
        
        # Data Augmentation: 50% chance to flip the image horizontally to make the model robust
        transforms.RandomHorizontalFlip(p=0.5), 
        
        # Critical Step: Converts PIL Image [H, W, C] to PyTorch Tensor [C, H, W]
        # and automatically scales pixel values from [0-255] down to [0.0 - 1.0]
        transforms.ToTensor(), 
        
        # Normalizes the values using specific mean and std deviations for each RGB channel
        # This shifts values from [0, 1] to roughly [-1, 1], which helps neural networks learn faster
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    print("--- 2. Instantiate Dataset ---")
    # Create dataset with 10 total fake images, passing in our transformation pipeline
    dataset = FakeImageDataset(num_samples=10, transform=custom_transform)
    print(f"Total dataset length: {len(dataset)}")
    
    # Manually testing __getitem__ to see what a single processed image looks like
    img, lbl = dataset[0]
    # Shape is [Channels, Height, Width] -> [3, 32, 32]
    print(f"Single Item Shape after transform: {img.shape}") 
    
    print("\n--- 3. Instantiate DataLoader ---")
    # DataLoader handles batching, shuffling, and multi-threaded loading securely
    dataloader = DataLoader(
        dataset=dataset,
        # Group every 4 separate images into a single thick tensor block for the GPU
        batch_size=4, 
        # Randomize order every epoch to prevent model bias based on data sequence
        shuffle=True, 
        # Increase this (e.g., 4 or 8) in production to load images via background CPU threads while the GPU trains
        num_workers=0 
    )
    
    print("\n--- 4. Iterating over Epochs ---")
    # Simulate a tiny training loop (2 full passes over the dataset)
    epochs = 2
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # The DataLoader automatically calls dataset.__getitem__() repeatedly
        # and stacks the results into the 'images' and 'labels' batch tensors!
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"  Batch {batch_idx+1}: Images shape: {images.shape} | Labels shape: {labels.shape}")
            # Target output:
            # Images shape is [BatchSize, Channels, Height, Width] == [4, 3, 32, 32]
            # Labels shape is [BatchSize] == [4]

if __name__ == "__main__":
    run_dataloader_demo()
