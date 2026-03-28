"""
Demo: Endpoint Deployment and Update
This script demonstrates the process of deploying a PyTorchModel to a 
SageMaker Real-Time Endpoint, invoking it with a payload, and updating the endpoint.

Note: Since we are running outside an active AWS environment with real billing,
the deployment functions are mocked to show the logic flow without incurring charges.
"""

from unittest.mock import MagicMock
import json

# Normally: from sagemaker.pytorch import PyTorchModel
# Normally: from sagemaker.serializers import JSONSerializer
# Normally: from sagemaker.deserializers import JSONDeserializer

class MockPredictor:
    """A mock representation of SageMaker's Predictor object which handles HTTP calls."""
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.status = "InService"
        
    def predict(self, payload):
        """Simulates sending a POST request to the live AWS endpoint."""
        print(f"\n[Network] Sending POST request to {self.endpoint_name}...")
        print(f"[Network] Payload size: {len(str(payload))} bytes")
        # Simulating the remote inference.py returning a calculated result
        return {"predictions": [0.85, 0.15], "predicted_class": "Cat"}
        
    def update_endpoint(self, initial_instance_count, instance_type):
        """Simulates requesting AWS to resize the server fleet without taking the service offline."""
        print(f"\n[AWS Infrastructure] Updating Endpoint '{self.endpoint_name}'...")
        print(f"[AWS Infrastructure] New Configuration: {initial_instance_count}x {instance_type}")
        print("[AWS Infrastructure] Performing Blue/Green deployment to prevent downtime...")
        # AWS provisions new servers first, routes traffic to them, then kills the old servers safely.
        print("[AWS Infrastructure] ✅ Update Complete. Traffic shifted to new instances.")
        
    def delete_endpoint(self):
        """Simulates tearing down the servers to stop accumulating hourly charges."""
        print(f"\n[AWS Infrastructure] Tearing down Endpoint '{self.endpoint_name}'...")
        self.status = "Deleted"
        print("[AWS Infrastructure] ✅ Instances terminated. Billing stopped.")

class MockPyTorchModel:
    """A mock representation of SageMaker's PyTorchModel artifact blueprint."""
    def __init__(self, model_data, role, entry_point, framework_version, py_version):
        self.model_data = model_data               # S3 path to the .tar.gz saved from training
        self.role = role                           # IAM Permissions
        self.entry_point = entry_point             # The inference.py script controlling request logic
        self.framework_version = framework_version # Specific PyTorch version required
        self.py_version = py_version               # Python version
        
    def deploy(self, initial_instance_count, instance_type, endpoint_name, serializer=None, deserializer=None):
        """Simulates commanding AWS to rent EC2 instances, load Docker, and start the web server."""
        print("\n--- AWS Deployment Initiated ---")
        print(f"Artifact: {self.model_data}")
        print(f"Inference Script: {self.entry_point}")
        print(f"Provisioning {initial_instance_count}x '{instance_type}' instances...")
        print("Booting AWS Deep Learning Containers...")
        print(f"Endpoint '{endpoint_name}' is now InService.")
        # Deploy returns a Predictor object wrapped around the URL
        return MockPredictor(endpoint_name)

def demonstrate_endpoint_lifecycle():
    print("--- SageMaker Endpoint Deployment Lifecycle ---")
    
    # 1. Define the Model Structure
    print("Configuring the PyTorchModel object...")
    pytorch_model = MockPyTorchModel(
        model_data="s3://my-cloud-bucket/models/resnet-v1/model.tar.gz",
        role="arn:aws:iam::123456789:role/SageMakerExecutionRole",
        entry_point="inference.py",
        framework_version="2.0.0",
        py_version="py310"
    )
    
    # 2. Deploy the Model
    # We use CPU instances (ml.m5.large) for basic inference to save costs. GPUs are rarely needed 24/7.
    print("\nDeploying to Real-Time Endpoint...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="prod-image-classifier-v1",
        serializer="JSONSerializer()",      # Automatically serialize local Python dicts to JSON before sending
        deserializer="JSONDeserializer()"  # Automatically parse returned JSON back to Python dicts
    )
    
    # 3. Simulate Client Invocation
    print("\n--- Simulating Client Web Application ---")
    # A dummy flattened image payload representing pixels
    sample_data = {"inputs": [0.5] * 224 * 224 * 3} 
    
    # Call the predictor (Acts like a Requests.post() call)
    response = predictor.predict(sample_data)
    print(f"[Client] Response Received: {json.dumps(response, indent=2)}")
    
    # 4. Simulate Auto-scaling / Updating
    print("\n--- Managing Production Infrastructure ---")
    print("Scenario: A marketing campaign launched. Traffic spiked 10x.")
    
    # Update the endpoint seamlessly to handle more traffic
    predictor.update_endpoint(
        initial_instance_count=3, # Scale up to 3 machines!
        instance_type="ml.m5.xlarge" # Upgrade to larger machines!
    )
    
    # 5. Cleanup
    print("\nWARNING: Always delete idle endpoints. You are billed per minute the server is running.")
    predictor.delete_endpoint()
    
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_endpoint_lifecycle()
