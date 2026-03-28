# SageMaker Estimators and Training

## Learning Objectives

- Adapt standard local PyTorch Scripts to ingest dynamic cloud environment variables.
- Configure and instantiate the SageMaker `Estimator` Class via the Python SDK.
- Pass Hyperparameters and custom Metrics to remote cloud instances.
- Handle external libraries and structural Dependencies using `requirements.txt`.
- Launch asynchronous Training Jobs and monitor their lifecycle.

## Why This Matters

Launching a Training Job in SageMaker means handing over control of your execution to a remote server. You can no longer hardcode file paths like `C:/data/images/`. The Estimator requires your code to dynamically read environment variables injected by SageMaker. Understanding how to pass hyperparameters, external Python packages, and monitor the logs remotely is the crux of modern MLOps training.

> **Key Term - MLOps (Machine Learning Operations):** The discipline of applying DevOps and software engineering best practices to the machine learning lifecycle. MLOps encompasses automated training pipelines, model versioning, deployment monitoring, and data drift detection — treating ML as a reliable, repeatable engineering process rather than a one-off experiment.

## The Concept

### Adapting PyTorch Scripts for the Cloud

When SageMaker boots your script inside its container, it sets specific environment variables defining where the data was downloaded (`SM_CHANNEL_TRAIN`) and where you must save your final model (`SM_MODEL_DIR`). Your script must use `os.environ` or `argparse` to read these paths dynamically.

> **Key Term - Environment Variable:** A dynamic named value set at the OS level that can be read by running programs. Instead of hardcoding `path = '/home/user/data'` in your code, you read `path = os.environ['DATA_DIR']`. SageMaker uses environment variables to inject runtime paths into training scripts, ensuring the script works correctly regardless of which specific server it runs on.

### Handling Dependencies

Your local script might import `pandas` or `scikit-learn`. If the base SageMaker PyTorch container doesn't have these, your training job will crash immediately with a `ModuleNotFoundError`. By simply including a `requirements.txt` file alongside your entry point script, SageMaker will automatically run `pip install` before executing your code.

### Passing Hyperparameters and Launching

The `Estimator` object accepts a `hyperparameters` dictionary (e.g., `{'epochs': 10, 'lr': 0.001}`). When you call `.fit()`, four things happen asynchronously:

> **Key Term - Asynchronous Execution:** An operation where the calling code does not wait for the operation to complete before continuing. Calling `estimator.fit()` submits a job to AWS and immediately returns control to your laptop. You can close your laptop; the training job runs in the cloud and reports results when done. The opposite, synchronous execution, would freeze your laptop until training completes.

1. SageMaker uploads your local script to an S3 bucket.
2. It provisions the requested GPU EC2 instances.
3. It downloads the data from S3 to the instance.
4. It executes your script, passing the hyperparameters as command-line arguments.

## Code Example

```python
# --- train.py (The local script that runs in the cloud) ---
import argparse
import os
import torch
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SageMaker passes these automatically based on Estimator config
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # SageMaker native environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.train}")
    print(f"Training for {args.epochs} epochs at LR: {args.learning_rate}")
    
    # ... (Training Loop happens here) ...
    
    # The model MUST be saved to the model-dir to be retained!
    print("Saving model weights...")
    torch.save({}, os.path.join(args.model_dir, 'model.pth'))
```

```python
# --- launch_job.py (Runs on your laptop to start the process) ---
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role='arn:aws:iam::123:role/execution_role',
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.0',
    py_version='py310',
    # Passing the hyperparams!
    hyperparameters={
        'epochs': 50,
        'learning_rate': 0.005
    }
)

# Launch the job pointing to where the raw data lives in S3
estimator.fit({'train': 's3://my-bucket/training-images/'})
```

## Additional Resources

- [SageMaker Training Toolkit Environments](https://github.com/aws/sagemaker-training-toolkit)
- [Monitor and Analyze Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/monitor.html)
