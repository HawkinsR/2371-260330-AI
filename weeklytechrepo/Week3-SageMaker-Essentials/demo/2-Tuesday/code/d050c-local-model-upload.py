"""
SageMaker Essentials Demo (d050c) - Hybrid Workflow: Personal PC to Cloud
Goal: Demonstrate the transition from local training (Laptop) to cloud hosting (SageMaker).

PREREQUISITES FOR PERSONAL PC MODE:
1. Install AWS CLI and run 'aws configure'.
2. Install SageMaker SDK: 'pip install sagemaker torch'.
3. Ensure your local IAM user has 'AmazonS3FullAccess' and 'AmazonSageMakerFullAccess'.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import tarfile
import shutil
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ==========================================================
# CONFIGURATION: USER ACTION MAY BE REQUIRED
# ==========================================================

# 1. Provide your SageMaker Execution Role ARN here 
# (You can find this in the AWS Console under IAM -> Roles)
# If running in SageMaker Studio, this will be detected automatically.
CUSTOM_ROLE_ARN = "arn:aws:iam::407975137156:role/2371-SM-Execution-Test"

# 2. Local environment configuration
LOCAL_MODEL_DIR = 'local_model_export'
TAR_NAME = 'model.tar.gz'
DEPLOY_INSTANCE = 'ml.m5.large'

# ==========================================================
# STEP A: ON YOUR PERSONAL PC (OR JUPYTERLAB INSTANCE)
# ==========================================================
print("\n" + "="*50)
print(" STEP A: LOCAL PC TASKS ".center(50, "="))
print("="*50)

# Define the model architecture 
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

print("\n--- Phase 1: Training Model Locally ---")
# Simulate training on your laptop
x_local = torch.randn(100, 1)
y_local = 2 * x_local + 1 + 0.1 * torch.randn(100, 1)

local_model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(local_model.parameters(), lr=0.01)

print("Training locally for 50 epochs...")
for epoch in range(50):
    outputs = local_model(x_local)
    loss = criterion(outputs, y_local)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Local Training Complete. Sample Weight: {local_model.linear.weight.item():.3f}")

# Save the local weights
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
model_path = os.path.join(LOCAL_MODEL_DIR, 'model.pth')
torch.save(local_model.state_dict(), model_path)
print(f"Saved local weights to {model_path}")

print("\n--- Phase 2: Packaging for the Cloud ---")
# Manually bundle model.pth and code/inference.py
code_dir = os.path.join(LOCAL_MODEL_DIR, 'code')
os.makedirs(code_dir, exist_ok=True)

src_inference = 'src/inference.py'
if os.path.exists(src_inference):
    shutil.copy(src_inference, os.path.join(code_dir, 'inference.py'))
    print("Injected inference.py into the local bundle.")
else:
    print(f"ERROR: {src_inference} not found. Ensure this script is in the Week3/demo folder.")
    exit(1)

# Create the .tar.gz file
with tarfile.open(TAR_NAME, "w:gz") as tar:
    tar.add(model_path, arcname='model.pth')
    tar.add(code_dir, arcname='code')
print(f"Local package '{TAR_NAME}' is ready for upload.")

# ==========================================================
# STEP B: CONNECTING TO AWS
# ==========================================================
print("\n" + "="*50)
print(" STEP B: CONNECTING TO AWS ".center(50, "="))
print("="*50)

try:
    session = sagemaker.Session()
    # Try to get the role; falls back to manual ARN if not in Studio
    try:
        role = sagemaker.get_execution_role()
        print("Connected: Running in SageMaker Studio (Role detected automatically).")
    except (ValueError, RuntimeError):
        role = CUSTOM_ROLE_ARN
        print(f"Connected: Running on a Personal PC. Using manual Role ARN: {role[:30]}...")
    
    bucket = session.default_bucket()
    print(f"Using S3 Bucket: {bucket}")
except Exception as e:
    print(f"\nCONNECTION ERROR: Could not connect to AWS. Did you run 'aws configure'?")
    print(f"Details: {e}")
    exit(1)

print("\n--- Phase 3: Uploading Local Artifacts to S3 ---")
s3_prefix = 'd050c-local-to-cloud-byom'
s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)
print(f"Local model uploaded! S3 Path: {s3_model_path}")

# ==========================================================
# STEP C: IN THE SAGEMAKER CLOUD
# ==========================================================
print("\n" + "="*50)
print(" STEP C: CLOUD DEPLOYMENT ".center(50, "="))
print("="*50)

print("\n--- Phase 4: Standing up the Hosted Endpoint ---")
pytorch_model = PyTorchModel(
    model_data=s3_model_path,
    role=role,
    sagemaker_session=session,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py'
)

print(f"Deploying to {DEPLOY_INSTANCE} (this will take several minutes)...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=DEPLOY_INSTANCE,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

print("\n--- Phase 5: Testing the Hybrid Bridge ---")
test_input = [10.0, 20.0, -5.0]
print(f"Sending test input from PC: {test_input}")

try:
    response = predictor.predict(test_input)
    print(f"Prediction from Cloud Endpoint: {response}")
    print("\nSUCCESS: You have successfully bridged the gap from local PC to Cloud Service!")
except Exception as e:
    print(f"Error during prediction: {e}")

# Cleanup Reminder
print("\n" + "="*40)
print("FINISHING UP")
print(f"To avoid cloud charges, run: predictor.delete_endpoint()")
print("="*40)
