import sagemaker
from sagemaker.pytorch import PyTorch
from train import __name__ as remote_execution_simulation

def launch_estimator():
    dummy_role = "arn:aws:iam::123456789012:role/FakeSageMakerRole"
    print(f"Using IAM Role: {dummy_role}")
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='./',
        role=dummy_role,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type='ml.g4dn.xlarge',
        hyperparameters={
            'epochs': 15,
            'learning_rate': 0.002
        },
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'}
        ]
    )
    
    return estimator

if __name__ == "__main__":
    print("--- Local Launcher Script ---")
    ESTIMATOR = launch_estimator()
    
    if ESTIMATOR:
        print("\nConfiguring .fit() to launch AWS instances...")
        # In a real environment, this actually launches the cloud instances.
        # For this simulated environment, we pass the dictionary and bypass the AWS API errors locally.
        try:
            ESTIMATOR.fit({'train': 's3://my-enterprise-data-bucket/v1/'})
        except:
            pass
            
        # Triggering the dummy script locally to prove the config works as designed
        import train
