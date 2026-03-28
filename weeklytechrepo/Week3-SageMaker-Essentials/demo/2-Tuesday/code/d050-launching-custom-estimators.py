"""
Demo: Launching Custom Estimators
This script mimics the separation between the local launcher script and the 
remote training script. We will trigger an 'asynchronous' training job 
using the PyTorch Estimator SDK, passing hyperparameters and simulating dependency injection.
"""

import os
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch

# =====================================================================
# FILE 1: train.py (Simulated remote execution)
# =====================================================================
def simulated_remote_training_script():
    """
    This simulates the entry_point script that runs ON the AWS GPU instance.
    """
    print("\n" + "="*50)
    print(">>> AWS SAGEMAKER CONTAINER INITIALIZING <<<")
    print("="*50)
    
    # Simulate SageMaker injecting environment variables automatically
    # /opt/ml/ is the standard SageMaker root path inside their official Docker containers
    os.environ['SM_MODEL_DIR'] = '/opt/ml/model' # Drop final artifacts here
    os.environ['SM_CHANNEL_TRAIN'] = '/opt/ml/input/data/train' # S3 data is downloaded here
    os.environ['SM_CURRENT_HOST'] = 'algo-1' # Server name (useful for multi-node distributed training)
    os.environ['SM_NUM_GPUS'] = '1' # GPU count detector
    
    parser = argparse.ArgumentParser()
    
    # 1. Custom Hyperparameters passed from the local Estimator API
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='sgd')
    
    # 2. Native SageMaker Environment Variables explicitly fetched
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # Simulating the exact CLI arguments SageMaker would pass when executing this internally
    # Like: python train.py --epochs 50 --learning_rate 0.005 --optimizer adam
    mock_sys_args = ['--epochs', '50', '--learning_rate', '0.005', '--optimizer', 'adam']
    args = parser.parse_args(mock_sys_args)
    
    print(f"[Container] Host: {os.environ['SM_CURRENT_HOST']} | GPUs available: {os.environ['SM_NUM_GPUS']}")
    print(f"[Container] Loading dataset from dynamically injected path: {args.train}")
    print(f"[Container] Hyperparameters detected -> Epochs: {args.epochs}, LR: {args.learning_rate}, Opt: {args.optimizer}")
    
    # If a requirements.txt file exists in the source_dir, SageMaker automatically runs pip install
    # BEFORE running this train.py script.
    print("[Container] Installing dependencies from requirements.txt... (Success)")
    
    print("\n[Container] Starting Training Loop...")
    for epoch in range(1, 4):
        # We simulate the loss output. Note the exact format "Loss: X.X"
        print(f"[Container] Epoch {epoch}/{args.epochs} - Loss: {1.0 / epoch:.4f}")
    print("[Container] ... Skipping remaining epochs for demo ...")
    
    # Saving exactly to args.model_dir is mandatory for AWS to capture the result
    print(f"\n[Container] Training Complete. Saving final model.pth to: {args.model_dir}")
    print(f"[Container] AWS will automatically compress this directory into model.tar.gz and upload it to S3.")
    print("="*50 + "\n")

# =====================================================================
# FILE 2: launch_job.py (Simulated local execution)
# =====================================================================
def demonstrate_estimator_launch():
    """
    This simulates what the Data Scientist runs on their local laptop or Studio Notebook
    to command AWS to spin up infrastructure and run train.py.
    """
    print("--- Local Laptop: Configuring Training Job ---")
    
    try:
        # Request AWS Identity Access Management (IAM) permissions payload
        role = sagemaker.get_execution_role()
    except ValueError:
        role = "arn:aws:iam::111122223333:role/SageMakerRole"
        
    print(f"Using IAM Role: {role}")
    
    # 1. Define the Estimator
    print("\nInstantiating PyTorch Estimator...")
    estimator = PyTorch(
        entry_point='train.py',                 # The script simulating the block above
        source_dir='./src',                     # Folder holding train.py and requirements.txt
        role=role,
        framework_version='2.0.0',              # AWS will pull the matching optimized Docker image
        py_version='py310',
        instance_count=1,
        instance_type='ml.p3.2xlarge',          # Requesting an expensive Nvidia V100 GPU instance!
        hyperparameters={
            'epochs': 50,
            'learning_rate': 0.005,
            'optimizer': 'adam'
        },
        # metric_definitions tells SageMaker what Regular Expression to look for in the standard output (print statements).
        # AWS scans the console text, extracts the numbers, and streams them to CloudWatch graphs live!
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'}
        ]
    )
    
    # 2. Launch the Job
    # .fit() handles the heavy lifting: Server provisioning, Docker pulling, S3 downloading, and Script running.
    print("\nLaunching Training Job...")
    print("Running: estimator.fit({'train': 's3://my-org-bucket/dataset-v1/'})")
    
    # In a real scenario, this is where the local script hangs/waits while AWS provisions servers remotely.
    # We will simulate the remote container execution here.
    simulated_remote_training_script()
    
    print("--- Local Laptop: Training Job Complete ---")
    print("The model artifact is now safely stored in S3 and the ml.p3.2xlarge instance has been terminated to stop billing.")
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_estimator_launch()
