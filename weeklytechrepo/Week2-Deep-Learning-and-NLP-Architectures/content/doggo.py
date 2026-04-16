import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

DATA_ROOT = "./../../W6/Demo/data/doggos"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LOG_DIR = "runs/doggo_logs"
MODEL_PATH = "doggo.pth"
NUM_EPOCHS = 10
PATIENCE = 20

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'doggo' folder is extracted in the script's directory.")
    sys.exit(1)

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class PreTrainedModel(nn.Module):
    def __init__(self):
        super(PreTrainedModel, self).__init__()

        # Load ResNet18 with weights pre-trained on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final layer with one that matches our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

class DoggoModel(nn.Module):
    def __init__(self):
        super(DoggoModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            # nn.Conv2d is a 2D convolution layer, slides a kernel over the input image
            # nn.MaxPool2d is a 2D max pooling layer, reduces the spatial dimensions of the input image
            # nn.ReLU is a rectified linear unit (ReLU) activation function, introduces non-linearity

            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
            nn.Conv2d(3, 16, kernel_size=3, stride=1), # Convolute 3x3 kernel, stepping by 1
            nn.ReLU(), 
            nn.MaxPool2d(2), # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128 -> 64x64
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # 32 filters, 62x62 pixels, 128 neurons
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2) # 128 neurons, 2 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False, True # should_stop=False, is_better=True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, False # should_stop=True/False, is_better=False

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device, early_stopping):
    print()

    print(f"\n--- Training Epoch {epoch+1} ---")
    
    model.train()
    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch + (epoch * len(dataloader)))
        
        print(f"Batch {batch}: Loss = {loss.item():>7f}")

        should_stop, is_better = early_stopping(loss.item())

        if is_better:
            print("New best model found! Loss: ", loss.item(), " Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, MODEL_PATH)

        if should_stop:
            print(f"\n--- Early stopping at batch {batch} (total patience: {early_stopping.patience}) ---")
            return model, early_stopping.best_loss, True

    end_time = time.time()
    print(f"Epoch {epoch+1} completed: {batch+1} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, early_stopping.best_loss, False

def evaluate(dataloader, model, loss_fn, writer, device):
    print()
    print("--- Eval Model ---")

    test_loss, correct, total= 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            if batch == 9: break
    
    writer.add_scalar("Loss/test", test_loss / total)
    
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Evaluation: Accuracy = {int(100 * correct / total)}%" )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
#   model = DoggoModel()
    model = PreTrainedModel().to(device)
    best_loss = float('inf')
    
    print("Adding graph to tensorboard...")
    dummy_data = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_data)

    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=PATIENCE)

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        early_stopping.best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model, current_best_loss, early_stop = train_loop(train_loader, model, criterion, optimizer, epoch, writer, device, early_stopping)
        evaluate(test_loader, model, criterion, writer, device)
        if early_stop:
            print(f"Training stopped early due to no improvement in loss over {PATIENCE} batches.")
            break

if __name__ == "__main__":
    main()