"""
Demo: Launching Custom Estimators (Live)
This script demonstrates the interaction between the local launcher and the 
remote training job. We use the PyTorch Estimator SDK to provision 
compute resources and execute a training script in the cloud.
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import time

def demonstrate_react_prompting():
    """
    Demo: ReAct Prompting Pattern (Reasoning + Acting)
    """
    print("--- ReAct Prompting Demo: Reasoning + Acting ---")
    
    react_sequence = [
        {
            "step": 1,
            "thought": "I need to train a model, but I don't want to use local compute. I will use a SageMaker Estimator.",
            "action": "LaunchEstimator[entry_point=train.py, instance=ml.m5.large, epochs=5]",
            "observation": "(Pending — .fit() will initiate a real training job)"
        }
    ]

    for item in react_sequence:
        print(f"--- Step {item['step']} ---")
        print(f"  \U0001f4ad Thought     : {item['thought']}")
        print(f"  \u26a1 Action      : {item['action']}")
        print(f"  \U0001f441\ufe0f  Observation : {item['observation']}")
        print()

def demonstrate_estimator_launch():
    """
    Demonstrates launching a real SageMaker Training Job.
    """
    print("--- SageMaker SDK: Configuring Live Training Job ---")
    
    # 1. Setup Session and Role
    session = sagemaker.Session()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        # Fallback for local testing (replace with a real role ARN if needed)
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        print("Note: Running outside SageMaker. Real .fit() requires valid credentials/role.")

    # 2. Define the Estimator
    print("\nInstantiating PyTorch Estimator...")
    estimator = PyTorch(
        entry_point='train.py',                 # Files in source_dir are bundled
        source_dir='src',                       # Directory containing train.py
        role=role,
        framework_version='2.0.0',              # AWS-optimized Docker image
        py_version='py310',
        instance_count=1,
        instance_type='ml.m5.large',            # CPU instance to keep costs low for the demo
        hyperparameters={
            'epochs': 5,
            'learning_rate': 0.01
        },
        # metric_definitions allows CloudWatch to parse logs and create graphs live
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'}
        ]
    )
    
    # 3. Launch the Job
    print("\n[Action] Launching Training Job via estimator.fit()...")
    print("In a real demo, this will provision a server, download your code/data, and run it.")
    
    # We use wait=False so the script doesn't hang in a live demo, 
    # but we'll print the link to the console for tracking.
    estimator.fit(wait=False)
    
    job_name = estimator.latest_training_job.name
    print(f"✅ Job Launched successfully: {job_name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?#/jobs/{job_name}")

if __name__ == "__main__":
    demonstrate_react_prompting()
    demonstrate_estimator_launch()
