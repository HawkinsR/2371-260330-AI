"""
Demo: Custom Inference Script
This script demonstrates the structure of a SageMaker 'inference.py' entry point. 
It defines the critical 'model_fn' to load the weights securely from a simulated 
tar.gz extraction into memory, and 'predict_fn' to route requests through the architecture.
"""

import os
import torch
import torch.nn as nn
import json

# =====================================================================
# SIMULATED ARCHITECTURE (Usually stored in model.py)
# =====================================================================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # A rudimentary layer taking 10 inputs and outputting 2 classes
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

# =====================================================================
# FILE: inference.py (The SageMaker Endpoints Server Script)
# =====================================================================

def model_fn(model_dir):
    """
    Hook 1/4: Loading Logic.
    SageMaker calls this ONCE when the Endpoint container starts up.
    It passes the directory where it extracted the S3 model.tar.gz file.
    """
    print(f"\n[Endpoint Server Start] Running `model_fn`...")
    print(f"Loading files from extracted directory: {model_dir}")
    
    # Automatically select GPU if the provisioned instance has one, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Assigning model to device: {device}")
    
    # 1. Instantiate the empty architecture framework
    model = SimpleClassifier()
    
    # 2. Load the raw weights from the .pth file
    # SageMaker extracts the model.tar.gz directly into `model_dir`
    model_path = os.path.join(model_dir, 'model.pth')
    
    # Simulate loading (in a real scenario, the file must exist directly alongside inference.py)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device, weights_only=True))
        print("Success: Weights loaded into architecture.")
    else:
        print(f"Warning: {model_path} not found. Proceeding with uninitialized weights for demo.")
        
    # 3. Mount to GPU/CPU and set to Eval Mode!
    # Eval mode is CRITICAL for inference to disable Dropout and freeze BatchNorm layers
    model.to(device)
    model.eval()
    
    # Check that gradients are disabled globally to save memory and ensure weights don't drift
    for param in model.parameters():
        param.requires_grad = False
        
    print("[Endpoint Server Ready] Model cached in memory.")
    # Return the fully initialized model object. SageMaker keeps this in memory.
    return model

def input_fn(request_body, request_content_type):
    """
    Hook 2/4: Parsing Logic.
    Converts incoming HTTP payload string into a PyTorch Tensor.
    """
    # SageMaker handles the HTTP server (like Flask/FastAPI) automatically
    # It just hands us the raw body and content type
    if request_content_type == 'application/json':
        print(f"\n[HTTP Request Received] Running `input_fn`...")
        print(f"Incoming JSON string: {request_body}")
        
        # Parse JSON string into a Python dictionary
        parsed = json.loads(request_body)
        # Convert the raw numbers into a PyTorch Float Tensor
        tensor = torch.tensor(parsed['inputs'], dtype=torch.float32)
        print(f"Converted to Tensor shape: {tensor.shape}")
        
        return tensor
    else:
        # Prevent the server from crashing mysteriously by explicitly rejecting bad formats
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_tensor, model):
    """
    Hook 3/4: Inference Logic.
    Takes the output from input_fn and the model from model_fn.
    """
    print(f"\n[Processing Engine] Running `predict_fn`...")
    
    # Ensure the incoming data is moved to the same device (GPU/CPU) as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    
    # We already called .eval() in model_fn, but using no_grad() is best practice
    # It completely disables the computation graph tracking, making inference much faster
    with torch.no_grad():
        prediction = model(input_tensor)
        
    print(f"Raw Output Tensor: {prediction}")
    return prediction

def output_fn(prediction_tensor, accept_content_type):
    """
    Hook 4/4: Formatting Logic.
    Converts the PyTorch Tensor back into a string for the HTTP response.
    """
    # The client requested JSON back
    if accept_content_type == 'application/json':
        print(f"\n[HTTP Response Out] Running `output_fn`...")
        
        # Convert the PyTorch Tensor back into standard Python lists using .tolist()
        output_list = prediction_tensor.tolist()
        
        # Serialize the Python dictionary into a raw JSON string
        response = json.dumps({'predictions': output_list})
        print(f"Final JSON Output to Client: {response}")
        print("-" * 50)
        
        return response, accept_content_type
    else:
        raise ValueError(f"Unsupported accept type: {accept_content_type}")

# =====================================================================
# SIMULATION ENGINE
# =====================================================================
def simulate_endpoint_lifecycle():
    print("--- Simulating SageMaker Endpoint Lifecycle ---")
    
    # Create fake model directory locally to simulate SageMaker extraction
    os.makedirs('./fake_model_dir', exist_ok=True)
    
    # Boot Sequence: SageMaker calls this once
    cached_model = model_fn('./fake_model_dir')
    
    # A client (e.g., Postman, Mobile App, or Web App) sends a POST request over the internet
    mock_http_body = '{"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}'
    mock_content_type = 'application/json'
    
    # Sequence of hooks fired sequentially by the SageMaker Container for EVERY request
    tensor_payload = input_fn(mock_http_body, mock_content_type)
    raw_prediction = predict_fn(tensor_payload, cached_model)
    final_http_response, _ = output_fn(raw_prediction, mock_content_type)
    
    print("\nSimulation Complete. This implies the client successfully received a response.")

if __name__ == "__main__":
    simulate_endpoint_lifecycle()
