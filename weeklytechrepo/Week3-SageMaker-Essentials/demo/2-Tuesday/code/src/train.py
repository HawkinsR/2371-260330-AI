import os
import argparse
import torch

def train():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # Sagemaker specific arguments. Path are set by Sagemaker.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()

    print(f"Training on Host: {os.environ.get('SM_CURRENT_HOST')}")
    print(f"Received Hyperparameters -> Epochs: {args.epochs}, LR: {args.learning_rate}")
    
    # Simulate a training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {1.0 / (epoch + 1):.4f}")
    
    # Save a dummy model artifact
    model_path = os.path.join(args.model_dir, 'model.pth')
    print(f"Saving dummy model to {model_path}")
    torch.save({"state": "demo"}, model_path)

if __name__ == '__main__':
    train()
