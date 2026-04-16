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
        # Simulated input: [store_id, day_of_week, temperature]
        self.fc = nn.Linear(3, 1) # Outputs predicted sales volume
        
    def forward(self, x):
        return self.fc(x)

# =====================================================================
# INFERENCE HOOKS (Your Task)
# =====================================================================
def model_fn(model_dir):
    """
    Called by SageMaker to load the model into memory.
    """
    print(f"\n[Endpoint Boot] Running `model_fn` from directory: {model_dir}")
    
    # 1. TODO: Determine device (cuda or cpu)
    device = None
    
    # 2. TODO: Instantiate an empty RetailForecaster
    model = None
    
    # 3. TODO: Construct the path to 'model.pth' using os.path.join and model_dir
    model_path = None

    # 4. TODO: Load the saved weights if the file exists.
    # In a real deployment this file is always present inside the extracted tar.gz.
    # For this simulation, guard with os.path.exists() so the script still runs.
    #   if os.path.exists(model_path):
    #       model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    #   else:
    #       print("(Simulating: model.pth not found — using random weights for lab purposes)")
    print("(Simulating the presence of model.pth — weights_only=True load skipped for lab)")

    # 5. TODO: Move the model to the defined device

    # 6. TODO: Set the model to evaluation mode

    
    return model

def input_fn(request_body, request_content_type):
    """
    Provided parsed input logic.
    """
    if request_content_type == 'application/json':
        parsed = json.loads(request_body)
        tensor = torch.tensor(parsed['features'], dtype=torch.float32)
        return tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Called by SageMaker to execute the prediction.
    """
    print(f"\n[Inference Engine] Running `predict_fn`...")
    
    # 1. TODO: Determine device and send input_data to it
    
    
    # 2. TODO: Wrap the forward pass in torch.no_grad()
    # Pass input_data through the model
    prediction = None
        
    return prediction

def output_fn(prediction_tensor, accept_content_type):
    """
    Provided output formatting logic.
    """
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
    
    # 1. Boot up
    loaded_model = model_fn(mock_model_dir)
    
    if loaded_model is not None:
        # 2. Client sends a request payload
        # Store #5, Tuesday (2), 75 Degrees
        mock_payload = '{"features": [[5.0, 2.0, 75.0]]}' 
        mock_content = 'application/json'
        
        # 3. Pipeline Execution
        try:
            tensor_in = input_fn(mock_payload, mock_content)
            raw_pred = predict_fn(tensor_in, loaded_model)
            final_res, _ = output_fn(raw_pred, mock_content)
        except Exception as e:
            print(f"Simulation failed during execution hooks: {e}")
    else:
        print("ERROR: model_fn returned None. Cannot proceed with simulation.")
