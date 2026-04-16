"""
Demo: Custom Training to Live Deployment (d050b)
==============================================
This demo walks through:
1. Training a real Linear Regression model on a SageMaker instance.
2. Saving the model artifact (.tar.gz) to S3.
3. Deploying that specific S3 artifact to a SageMaker Inference Endpoint.
4. Sending a live test request to the endpoint.

Usage:
------
Set USE_GPU = True to toggle into a more powerful GPU instance (g4dn).
"""

import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import torch
import time

# --- CONFIGURATION ---
# Set this to True to move from CPU to a more powerful GPU instance
USE_GPU = False 

TRAIN_INSTANCE = 'ml.g4dn.xlarge' if USE_GPU else 'ml.m5.large'
DEPLOY_INSTANCE = 'ml.m5.large' # Endpoint instance

print(f"--- Configuration: Training on {TRAIN_INSTANCE} ---")

def run_demo():
    # 1. Setup Session and Role
    session = sagemaker.Session()
    # Studio-native way to get the role
    role = sagemaker.get_execution_role()
    print(f"Using execution role: {role}")

    # 2. Phase 1: Training the Model
    # ... (estimator setup remains the same, but it now bundles the code)
    print("\n[Phase 1] Launching Training Job...")
    estimator = PyTorch(
        entry_point='train_linear.py',
        source_dir='src',
        role=role,
        sagemaker_session=session,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type=TRAIN_INSTANCE,
        hyperparameters={
            'epochs': 20,
            'learning_rate': 0.05
        }
    )

    print(f"Training script: src/train_linear.py")
    print("This will generate synthetic data (y=2x+1) and inject inference.py into the bundle.")
    
    # We wait for training to finish because we need the artifact for deployment
    print("\nWaiting for training to complete (this usually takes 3-5 mins)...")
    estimator.fit(wait=True)

    # 3. Interlude: Observe the Artifact
    model_data = estimator.model_data
    print(f"\n✅ Training Complete!")
    print(f"📦 Model Artifact saved to S3: {model_data}")

    # 4. Phase 2: Deployment
    print("\n[Phase 2] Deploying Model to SageMaker Endpoint...")
    print(f"Using estimator.deploy() for stable transition...")
    
    # NOTE: We use estimator.deploy() because it handles all the 
    # 'repacking' logic and S3 paths automatically, which is more 
    # robust than manually creating a Model object.
    
    # We still need to specify WHICH script inside 'src' is the translator (inference.py).
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=DEPLOY_INSTANCE,
        entry_point='inference.py',
        source_dir='src',
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    # 5. Phase 3: Validation
    print("\n[Phase 3] Testing the Live Endpoint...")
    
    # We expect y = 2x + 1
    test_input = [10.0, 20.0, -5.0]
    print(f"Sending test input: {test_input}")
    
    response = predictor.predict(test_input)
    print(f"Raw Response from Endpoint: {response}")
    
    print("\n--- Summary ---")
    for val, pred in zip(test_input, response):
        expected = 2 * val + 1
        print(f"Input: {val:4} | Predicted: {pred[0]:.4f} | Expected (~2x+1): {expected:.1f}")

    # 6. Cleanup Instructions
    print("\n" + "="*40)
    print("⚠️  IMPORTANT: COST ALERT")
    print(f"The endpoint '{predictor.endpoint_name}' is now RUNNING.")
    print("To avoid ongoing charges, run the following code when finished:")
    print(f"  >>> predictor.delete_endpoint()")
    print("="*40)

if __name__ == "__main__":
    run_demo()
