"""
Demo: SageMaker Script Mode Setup
This script demonstrates how to configure a SageMaker Estimator using the Python SDK 
and how a basic PyTorch training script should be structured for Script Mode.
"""

import os
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch

# --- Part 1: The Local Script Mode Entry Point (train.py simulation) ---
# In reality, this block of code would live in a separate file (e.g., source_dir/train.py)

def dummy_train_script():
    """
    This function simulates what goes inside the 'train.py' entry point.
    SageMaker automatically passes hyperparameters as command-line arguments.
    It also sets environment variables telling the script where data is located.
    """
    # argparse is the standard Python way to parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # SageMaker passes hyperparameters as arguments when launching the container
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # SageMaker native environment variables
    # SM_MODEL_DIR: This is the critical directory! Anything saved here is automatically 
    # compressed into model.tar.gz and uploaded to S3 when training finishes.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    
    # SM_CHANNEL_TRAIN: The directory inside the Docker container where SageMaker 
    # automatically downloaded the training dataset from S3 before running this script.
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    
    # parse_known_args is safer than parse_args because SageMaker sometimes adds internal args we don't care about
    args, _ = parser.parse_known_args()
    
    print("\n[INSIDE CONTAINER] Script Mode Execution Started!")
    print(f"[INSIDE CONTAINER] Hyperparameters - Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"[INSIDE CONTAINER] Training Data Directory: {args.train}")
    print(f"[INSIDE CONTAINER] Model Output Directory: {args.model_dir}")
    print("[INSIDE CONTAINER] Training complete. Saving artifact...\n")

# --- Part 2: The SageMaker Host SDK ---

def setup_sagemaker_estimator():
    print("--- SageMaker SDK Configuration ---")
    
    # 1. Establish the Session and IAM Role
    # An IAM Role is an AWS identity with permission policies. This role needs permission 
    # to spin up EC2 instances, read from S3 buckets, and write to CloudWatch logs.
    print("Configuring Execution Role and Session...")
    try:
        # This works if running inside a SageMaker Notebook / Studio / EC2 instance with an attached role
        role = sagemaker.get_execution_role()
    except ValueError:
        # Fallback for local execution simulation on a laptop without AWS credentials configured
        role = "arn:aws:iam::123456789012:role/FakeSageMakerExecutionRole"
        print(f"Local Environment Detected. Using dummy role: {role}")
        
    session = sagemaker.Session()
    
    # 2. Define the PyTorch Estimator
    print("\nDefining the PyTorch Estimator...")
    print("This tells AWS *how* to build the cluster and *what* code to inject into it.")
    
    # The Estimator is a blueprint. It doesn't actually launch anything until .fit() is called.
    estimator = PyTorch(
        entry_point='train.py',             # The specific Python script to execute first
        source_dir='./src',                 # The folder containing train.py AND requirements.txt
        role=role,                          # AWS IAM Permissions
        framework_version='2.0.0',          # Tells SageMaker which official PyTorch Docker Image to pull
        py_version='py310',                 # The Python version expected inside the container
        instance_count=1,                   # How many EC2 Virtual Machines to provision
        instance_type='ml.m5.xlarge',       # The exact hardware to rent (Cost-effective CPU for testing)
        hyperparameters={                   # These map directly to the argparse flags in Part 1!
            'epochs': 20,
            'learning-rate': 0.005,
            'batch-size': 128
        }
    )
    
    print("\nEstimator successfully configured.")
    print("To launch the cluster, download data, and run the code, you would call: estimator.fit({'train': 's3://my-bucket/training-data'})")
    print("-" * 50)

if __name__ == "__main__":
    setup_sagemaker_estimator()
    print("\n--- Simulating Container Execution ---")
    
    # Simulate the environment variables that SageMaker automatically injects into the Docker container
    os.environ['SM_MODEL_DIR'] = '/opt/ml/model'
    os.environ['SM_CHANNEL_TRAIN'] = '/opt/ml/input/data/train'
    
    # Execute the "remote" script locally for demo purposes
    dummy_train_script()
