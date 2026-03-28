import json
from unittest.mock import MagicMock

# =====================================================================
# MOCK AWS INFRASTRUCTURE (Do not edit this section)
# =====================================================================
class MockPredictor:
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.status = "InService"
        
    def predict(self, payload):
        print(f"\n[Network] Sending POST request to {self.endpoint_name}...")
        return {"predictions": [0.99, 0.01], "predicted_class": "Approved"}
        
    def delete_endpoint(self):
        print(f"\n[AWS Infrastructure] Tearing down Endpoint '{self.endpoint_name}'...")
        self.status = "Deleted"
        print("[AWS Infrastructure] ✅ Instances terminated. Billing stopped.")

class MockPyTorchModel:
    def __init__(self, model_data, role, entry_point, framework_version, py_version):
        self.model_data = model_data
        self.role = role
        self.entry_point = entry_point
        
    def deploy(self, initial_instance_count, instance_type, endpoint_name, serializer=None, deserializer=None):
        print("\n--- AWS Deployment Initiated ---")
        print(f"Artifact: {self.model_data}")
        print(f"Provisioning {initial_instance_count}x '{instance_type}' instances...")
        print("Booting AWS Deep Learning Containers...")
        print(f"Endpoint '{endpoint_name}' is now InService.")
        return MockPredictor(endpoint_name)

# =====================================================================
# YOUR TASKS
# =====================================================================
def deploy_production_model(iam_role):
    print("--- Configuring Production Endpoint ---")
    
    # 1. TODO: Configure the PyTorchModel
    model = None
    
    # 2. TODO: Call .deploy() on the model
    # Request 2 instances of 'ml.m5.large' named 'production-endpoint-v1'
    predictor = None
    
    return predictor

def test_and_cleanup(predictor):
    print("\n--- Testing Live Endpoint ---")
    dummy_payload = {"user_id": 12345, "transaction_amount": 500}
    
    # 1. TODO: Call .predict() on the predictor object passing the payload
    response = None
    
    if response:
        print(f"[Client] Success! Response Received: {json.dumps(response, indent=2)}")
        
    # 2. TODO: Tear down the endpoint to stop billing!
    
    

if __name__ == "__main__":
    ROLE = "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    
    live_predictor = deploy_production_model(ROLE)
    
    if live_predictor:
        test_and_cleanup(live_predictor)
    else:
        print("ERROR: Predictor returning None. Finish the deploy_production_model function.")
