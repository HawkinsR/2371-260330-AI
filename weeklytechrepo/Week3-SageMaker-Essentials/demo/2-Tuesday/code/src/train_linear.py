import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the model architecture
# This MUST match the definition in inference.py
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def train():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    
    args, _ = parser.parse_known_args()

    # Determine device (CPU vs GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Generate Synthetic Data (y = 2x + 1)
    print("Generating synthetic linear data...")
    x = torch.randn(100, 1).to(device)
    y = 2 * x + 1 + 0.1 * torch.randn(100, 1).to(device)

    # 2. Define simple linear model
    model = LinearModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # 3. Training Loop
    print(f"Starting training loop ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")
    
    # 4. Save Model 
    # Important: SageMaker expects the model artifact at SM_MODEL_DIR
    model_path = os.path.join(args.model_dir, 'model.pth')
    print(f"Saving model state to {model_path}")
    
    # Extract weight and bias for logging verification
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f"Final Model: y = {w:.3f}x + {b:.3f}")
    
    torch.save(model.state_dict(), model_path)

    # 5. [IMPROVEMENT] Bundling inference script with the model
    # This bypasses the need for "repacking" on the client side during deployment.
    code_dir = os.path.join(args.model_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    
    import shutil
    # The source_dir files are available in the current working directory of the container
    if os.path.exists('inference.py'):
        print(f"Injecting inference.py into {code_dir} for standalone deployment...")
        shutil.copy('inference.py', os.path.join(code_dir, 'inference.py'))
    else:
        print("Warning: inference.py not found in working directory. Deployment might require repacking.")

if __name__ == '__main__':
    train()
