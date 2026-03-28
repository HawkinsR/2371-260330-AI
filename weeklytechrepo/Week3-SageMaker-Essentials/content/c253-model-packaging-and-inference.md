# Model Artifacts and Inference

## Learning Objectives

- Define the structure and packaging logic of a SageMaker `model.tar.gz` artifact.
- Create an `inference.py` Entry Point script to handle real-time HTTP requests.
- Implement the `model_fn` hook to execute loading logic securely into memory.
- Implement the `predict_fn` hook to route payloads through inference architectures.
- Test Inference Capabilities locally before executing remote cloud deployments.

## Why This Matters

Training a model is useless if external applications (like a React web app) cannot send it data and receive a prediction. When SageMaker finishes training, it zips everything in the `SM_MODEL_DIR` into a `model.tar.gz` on S3. To serve it, we must provide SageMaker with a rigid, structured inference script defining exactly how to un-pickle that model into RAM and how to process incoming JSON payloads into PyTorch Tensors.

> **Key Term - Model Artifact:** The trained model's saved state — typically the weights (`.pth file`) and any architecture code needed to reconstruct the model. In SageMaker, training produces a `model.tar.gz` archive on S3 containing all necessary files to later deploy the model as a server.

> **Key Term - Serialization / JSON Payload:** Serialization converts a Python object (like a list of numbers) into a transmittable format like JSON. A payload is the data body of an HTTP request. When a web app sends a prediction request, it serializes the input data into a JSON string, transmits it to the model endpoint, and receives a JSON response back.

## The Concept

### The `model.tar.gz` Structure

SageMaker deployment containers pull down the `.tar.gz` artifact and extract it. Inside, it expects to find your saved `.pth` weights and any structural code needed (like your custom `nn.Module` class definitions). Without the architecture class, PyTorch cannot map the raw weights back into a usable forward pass.

### The `inference.py` Hooks

When SageMaker boots up an endpoint, it looks for an `inference.py` script containing specific reserved function names:

> **Key Term - Inference Hook:** A reserved function name (`model_fn`, `predict_fn`, etc.) that SageMaker calls automatically at specific moments in the serving lifecycle. These hooks act as a contract between your code and the SageMaker serving framework, defining what happens when a server starts, receives data, and returns predictions.

1. **`model_fn(model_dir)`:** Runs exactly once when the server starts. It loads the weights from disk into GPU/CPU memory and returns the instantiated model object.
2. **`input_fn(request_body, request_content_type)`:** Parses the incoming HTTP request (usually JSON) and converts it into a PyTorch Tensor.
3. **`predict_fn(input_data, model)`:** Takes the Tensor from the input hook, passes it precisely through the `model.forward()` pass, and returns the raw prediction.
4. **`output_fn(prediction, content_type)`:** Converts the PyTorch Tensor prediction back into JSON to be sent back to the client over HTTP.

*Note: SageMaker PyTorch containers have default implementations for `input_fn` and `output_fn`, but `model_fn` is strictly mandatory for BYOS.*

## Code Example

```python
# --- inference.py ---
import os
import torch
import json
from my_model_architecture import CustomResNet # Assuming this was packaged in the tar.gz

# 1. Loading Logic (Runs Once on Server Boot)
def model_fn(model_dir):
    """
    Called by SageMaker to load the model into memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CustomResNet()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device, weights_only=True))
        
    model.to(device)
    model.eval() # CRITICAL for inference!
    return model

# 2. Inference Logic (Runs on every HTTP Request)
def predict_fn(input_data, model):
    """
    Called by SageMaker to execute the prediction.
    input_data is already parsed into a Tensor by the default input_fn.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        prediction = model(input_data)
        
    return prediction
```

## Additional Resources

- [Deploy PyTorch Models on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models)
- [SageMaker Inference Handlers](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html)
