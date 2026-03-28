import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Simulating loading a PIL image from disk
        raw_image = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))
        
        if self.transform:
            raw_image = self.transform(raw_image)
            
        return raw_image, torch.tensor(label, dtype=torch.float32)


def build_pipeline():
    print("--- Building Data Pipeline ---")
    
    # Mock data references
    mock_paths = [f"part_{i}.jpg" for i in range(100)] # 100 images
    mock_labels = [np.random.randint(0, 2) for _ in range(100)] # 0 or 1
    
    # 1. Define the TorchVision Transforms
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. Instantiate the Custom Dataset
    dataset = DefectDataset(mock_paths, mock_labels, transform=data_transform)
    print(f"Dataset securely loaded with {len(dataset)} items.")
    
    # 3. Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("\n--- Iterating Pipeline ---")
    # 4. Iterate over the DataLoader for one "epoch"
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} - Images Shape: {images.shape}")
        # Early exit just for output brevity
        if batch_idx == 1:
            break

if __name__ == "__main__":
    build_pipeline()
