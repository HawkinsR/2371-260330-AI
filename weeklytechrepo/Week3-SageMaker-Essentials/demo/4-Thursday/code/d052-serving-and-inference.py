"""
Demo: Serving and Inference (Live)
This script demonstrates how to host a custom PyTorch model on a 
SageMaker Real-time Endpoint using the 'inference.py' hook pattern.
"""

import os
import torch
import torch.nn as nn
import json
import tarfile
import sagemaker
from sagemaker.pytorch.model import PyTorchModel

# 1. Custom Architecture
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

# =====================================================================
# SAGEMAKER INFERENCE HOOKS (Must be in a standalone script file)
# =====================================================================
# For the demo, we assume these hooks are saved in 'inference.py'
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device, weights_only=True))
    return model.to(device).eval()

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(input_tensor)

# =====================================================================
# LIVE DEPLOYMENT ENGINE
# =====================================================================
def run_live_serving_demo():
    print("--- SageMaker Serving: Live Endpoint Deployment ---")
    
    # 1. Prepare Artifacts
    # In a real scenario, these come from your training job in S3.
    # Here, we create a dummy model file and tar it.
    os.makedirs('model', exist_ok=True)
    torch.save(SimpleClassifier().state_dict(), 'model/model.pth')
    
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('model/model.pth', arcname='model.pth')
    
    # 2. Setup Session
    session = sagemaker.Session()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

    # 3. Define the Model object
    print("\nDefining PyTorch Model...")
    model = PyTorchModel(
        model_data='model.tar.gz',
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        entry_point=__file__ # In this demo, the script itself contains the hooks
    )

    predictor = None
    try:
        # 4. Deploy to Endpoint
        print("\n[Action] Deploying to Real-time Endpoint (ml.t2.medium)...")
        print("Note: This provisions real hardware and takes ~5 minutes.")
        
        # In a live lecture, the instructor might want to run this and come back later.
        # predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
        
        print("\nDemo code for deployment:")
        print(">>> predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')")
        
        # 5. Inference Simulation (once live)
        example_input = [[1.0]*10]
        print(f"\nInference Example:\n>>> response = predictor.predict({example_input})")

    except Exception as e:
        print(f"❌ Error during deployment setup: {e}")
    finally:
        # Cleanup
        if predictor:
            print("\n[Cleanup] Deleting endpoint...")
            predictor.delete_endpoint()
        
        # Local file cleanup
        if os.path.exists('model/model.pth'): os.remove('model/model.pth')
        if os.path.exists('model'): os.rmdir('model')
        if os.path.exists('model.tar.gz'): os.remove('model.tar.gz')

if __name__ == "__main__":
    run_live_serving_demo()
