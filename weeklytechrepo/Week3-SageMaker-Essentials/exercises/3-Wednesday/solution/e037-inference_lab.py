import os
import torch
import torch.nn as nn
import json

# =====================================================================
# SIMULATED ARCHITECTURE
# =====================================================================
class RetailForecaster(nn.Module):
    def __init__(self):
        super(RetailForecaster, self).__init__()
        self.fc = nn.Linear(3, 1)
        
    def forward(self, x):
        return self.fc(x)

# =====================================================================
# INFERENCE HOOKS (Your Task)
# =====================================================================
def model_fn(model_dir):
    print(f"\n[Endpoint Boot] Running `model_fn` from directory: {model_dir}")
    
    # 1. Determine device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Instantiate an empty RetailForecaster
    model = RetailForecaster()
    
    # 3. Construct the path to 'model.pth'
    model_path = os.path.join(model_dir, 'model.pth')
    
    print("(Simulating the presence of model.pth...)")
    
    # 4. Move the model to the defined device
    model.to(device)
    
    # 5. Set the model to evaluation mode
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        parsed = json.loads(request_body)
        tensor = torch.tensor(parsed['features'], dtype=torch.float32)
        return tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    print(f"\n[Inference Engine] Running `predict_fn`...")
    
    # 1. Determine device and send input_data to it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    # 2. Wrap the forward pass in torch.no_grad()
    with torch.no_grad():
        prediction = model(input_data)
        
    return prediction

def output_fn(prediction_tensor, accept_content_type):
    if accept_content_type == 'application/json':
        output_list = prediction_tensor.tolist()
        response = json.dumps({'predicted_sales': output_list})
        print(f"Final JSON Output to Client: {response}\n")
        return response, accept_content_type
    else:
        raise ValueError(f"Unsupported accept type: {accept_content_type}")

# =====================================================================
# SIMULATION ENGINE
# =====================================================================
if __name__ == "__main__":
    print("--- Simulating Endpoint Invocation ---")
    mock_model_dir = './extracted_tar_dir'
    
    loaded_model = model_fn(mock_model_dir)
    
    if loaded_model is not None:
        mock_payload = '{"features": [[5.0, 2.0, 75.0]]}' 
        mock_content = 'application/json'
        
        try:
            tensor_in = input_fn(mock_payload, mock_content)
            raw_pred = predict_fn(tensor_in, loaded_model)
            final_res, _ = output_fn(raw_pred, mock_content)
        except Exception as e:
            print(f"Simulation failed during execution hooks: {e}")
    else:
        print("ERROR: model_fn returned None. Cannot proceed with simulation.")
