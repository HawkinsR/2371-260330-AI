import os
import torch
import torch.nn as nn
import json

# Define the same model architecture as used in training
# Note: SageMaker needs to know what the 'house' (architecture) looks like 
# before it can move the 'furniture' (trained weights) into it.
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def model_fn(model_dir):
    """
    1. THE KEY: model_fn loads the model artifact from disk.
    SageMaker calls this once when the endpoint container starts.
    """
    print(f"Loading model from {model_dir}...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LinearModel()
        
        model_path = os.path.join(model_dir, 'model.pth')
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))
        
        print("Model loaded successfully.")
        return model.to(device)
    except Exception as e:
        print(f"EXCEPTION during model_fn: {e}")
        # We re-raise the exception so SageMaker knows the loading failed
        raise e

def input_fn(request_body, request_content_type):
    """
    2. THE RECEPTIONIST: input_fn converts raw requests into Tensors.
    This runs on every incoming request.
    """
    print(f"Receiving request of type: {request_content_type}")
    if request_content_type == 'application/json':
        # Decode bytes to string if necessary
        body = request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body
        data = json.loads(body)
        # Expecting a list of floats, e.g., [1.0, 2.0, 3.0]
        return torch.tensor(data, dtype=torch.float32).view(-1, 1)
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    3. THE BRAIN: predict_fn performs the actual math.
    It takes the data from input_fn and uses the model from model_fn.
    """
    print(f"Performing inference on input shape: {input_data.shape}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_data = input_data.to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(input_data)
        
        print("Inference successful.")
        return prediction.cpu().numpy()
    except Exception as e:
        print(f"EXCEPTION during predict_fn: {e}")
        raise e

def output_fn(prediction, content_type):
    """
    4. THE MESSENGER: output_fn turns Tensors back into JSON for the user.
    """
    print("Serializing output...")
    # NOTE: We return just the string. The SageMaker framework handles 
    # the HTTP headers for us. Returning a tuple (data, type) can 
    # sometimes cause worker crashes in certain PyTorch versions.
    return json.dumps(prediction.tolist())
