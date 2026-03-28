import os
import argparse

def train_model(args):
    """
    Simulates a PyTorch training loop utilizing the parsed arguments.
    """
    print("--- Starting Cloud Training Run ---")
    
    # 1. Load Data
    print(f"Loading training data from: {args.train}")
    
    # 2. Setup Loop
    print(f"Training with epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.learning_rate:.4f}")
    
    # 3. Save Model Artifact
    print(f"Saving final model artifact to: {args.model_dir}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Add hyperparameter arguments
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # Add SageMaker Environment Variable arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    
    # Parse the known arguments
    args, _ = parser.parse_known_args()
    
    # Call train_model with the parsed arguments
    train_model(args)
