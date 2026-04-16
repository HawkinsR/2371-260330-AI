# Deployment and Inference Scripts

## Learning Objectives

- Deploy Real-time Inference Endpoints using the SageMaker Python SDK.
- Select appropriate Instance Types for inference latency and throughput requirements.
- Implement `model_fn` and `predict_fn` logic for custom serving containers.
- Configure Auto-scaling to handle fluctuating request volumes automatically.
- Establish Approval Workflows and Versioning for production deployments.

Training a model is useless if external applications cannot consume its predictions. Transitioning from a trained artifact (`model.tar.gz`) to a live, persistent **Real-time Endpoint** is the final step in the ML pipeline. Understanding how to write custom inference scripts and configure the underlying hardware ensures your model is responsive, cost-effective, and ready for high-traffic production environments.

> **Key Term - Model Artifact:** The trained model's saved state — typically the weights (`.pth file`) and any architecture code needed to reconstruct the model. In SageMaker, training produces a `model.tar.gz` archive on S3 containing all necessary files to later deploy the model as a server.

> **Key Term - Serialization / JSON Payload:** Serialization converts a Python object (like a list of numbers) into a transmittable format like JSON. A payload is the data body of an HTTP request. When a web app sends a prediction request, it serializes the input data into a JSON string, transmits it to the model endpoint, and receives a JSON response back.

### Real-time Inference Endpoints

A SageMaker Endpoint is a dedicated cloud server (or cluster of servers) running a web service that hosts your model. It provides a unique URL where you can send POST requests containing data.

Unlike training jobs, which are ephemeral (they shut down after training), endpoints are **persistent**. They stay running until you explicitly delete them, ensuring your application always has access to the model.

> **Key Term - Real-time Endpoint:** A persistent, managed HTTPS URL provided by SageMaker that hosts a model and serves predictions with low latency.

> **Key Term - Instance Selection (Inference):** The process of choosing the right hardware (CPU vs. GPU, memory size) for serving. For small models, `ml.t2.medium` (CPU) might suffice. For large LLMs or computer vision, `ml.g4dn` or `ml.g5` (GPU) instances are often necessary for sub-second response times.

### Auto-scaling and Approval

In production, traffic is never constant. **Auto-scaling** allows SageMaker to automatically increase the number of instances during peak hours and decrease them during quiet periods, balancing performance and cost.

Furthermore, before an endpoint is updated with a new model version, it should pass through an **Approval Workflow**. This ensures a human or an automated test suite has verified the model's accuracy and safety before it touches live users.

> **Key Term - Auto-scaling:** An AWS feature that dynamically adjusts the number of instances in an endpoint cluster based on metrics like "Invocations Per Instance" or "CPU Utilization."

> **Key Term - Approval Workflow:** A gated process in the MLOps lifecycle where a model must be explicitly marked as "Approved" in the Model Registry before it can be deployed to a production endpoint.

```python
import boto3

autoscaling = boto3.client('application-autoscaling', region_name='us-east-1')

endpoint_name = 'my-production-endpoint'
resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'

# Step 1: Register the endpoint variant as a scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=resource_id,
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,   # Never scale below 1 instance
    MaxCapacity=4    # Never scale above 4 instances
)

# Step 2: Create a Target Tracking policy
# Triggers scale-out when each instance handles > 100 invocations/minute
autoscaling.put_scaling_policy(
    PolicyName='InvocationsPerInstanceScaling',
    ServiceNamespace='sagemaker',
    ResourceId=resource_id,
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 100.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,   # Wait 5 min before scaling in (removing instances)
        'ScaleOutCooldown': 60    # Scale out quickly (within 1 min) when load spikes
    }
)
print("Auto-scaling policy registered successfully.")
```

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
- [Automatically Scaling SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
