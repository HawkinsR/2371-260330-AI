# SageMaker Endpoints and Scaling

## Learning Objectives

- Architect Deployment Frameworks converting model artifacts into Real-time Endpoints.
- Assess Instance Selection criteria comparing CPU vs GPU vs AWS Inferentia hardware.
- Configure Auto-scaling policies to handle variable production traffic seamlessly.
- Implement robust procedures Updating & Rolling Back Endpoints without downtime.
- Evaluate Cost Optimization Basics aligning traffic elasticity with backend expenditure.

## Why This Matters

An inference script only works if it is securely hosted on an accessible network endpoint. Machine learning inference in production operates identically to a web server fielding HTTP requests. A data scientist must not only deploy the model but configure it to scale out servers dynamically when 1,000 users arrive at once, and shut servers down to save money when traffic sleeps. This separates local proofs-of-concept from enterprise-grade production software.

> **Key Term - Endpoint:** A URL address (e.g., `https://runtime.sagemaker.amazonaws.com/endpoints/my-model/invocations`) that listens for incoming HTTP requests containing data and returns predictions. It is the publicly accessible address by which other applications interact with your deployed model.

## The Concept

### Real-Time Endpoints and Instances

When you deploy to a SageMaker Real-Time Endpoint, AWS provisions an elastic load balancer backed by one or more EC2 instances hosting your `.tar.gz` and `inference.py` inside a Docker container.
Instance selection is vital.

- **CPU:** Cheap, but significantly slower for deep learning inference tasks (like Transformers) due to the lack of parallelized matrix multiplication hardware.
- **GPU (e.g., g4dn):** Expensive, but executes massive matrix multiplications instantly.
- **AWS Inferentia:** Custom silicon built by Amazon specifically for inference. It offers the best cost-to-performance ratio but requires compiling your model into a specific format using AWS Neuron SDKs.

### Auto-scaling and Updates

Hardcoding your endpoint to use exactly "3 GPU servers" forever guarantees you will either crash during a spike or waste money overnight. By attaching an Auto-scaling policy, CloudWatch monitors the "Invocations Per Instance" metric. It spins up identical replicas of your container under the load balancer when traffic spikes, and tears them down when idle.
When deploying version 2.0 of your model, SageMaker handles "Blue/Green Deployments." It boots the new model alongside the old one, shifts traffic transparently to the new model, and only kills the old model after health checks verify the new endpoints are returning 200 OK statuses.

> **Key Term - Auto-scaling:** A cloud capability that automatically adjusts the number of running server instances in response to live traffic. When many requests arrive simultaneously, auto-scaling spins up additional servers to share the load. When traffic decreases, it terminates unused servers to save cost.

> **Key Term - Blue/Green Deployment:** A zero-downtime release strategy where two identical production environments run simultaneously — the current version ("blue") serving live traffic and the new version ("green") being tested. After the green environment is verified healthy, traffic is switched from blue to green instantly, allowing instant rollback if issues arise.

## Code Example

```python
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# 1. Define the serving model using your trained S3 artifact and inference script
pytorch_model = PyTorchModel(
    model_data="s3://my-bucket/models/my-trained-model/model.tar.gz",
    role="arn:aws:iam::123:role/execution_role",
    entry_point="inference.py",    # The script containing model_fn and predict_fn
    framework_version="2.0",
    py_version="py310"
)

# 2. Deploy to a Real-Time Endpoint (This provisions the underlying EC2 server)
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",    # CPU instance, acceptable for simple models
    endpoint_name="my-production-classifier",
    serializer=JSONSerializer(),    # Automatically parses Python dicts to JSON
    deserializer=JSONDeserializer() # Parses JSON back to Python dicts
)

# 3. Inference (Simulating a frontend application calling the API)
dummy_payload = {"image_tensor_list": [...]} # Imagine raw pixel data here
response = predictor.predict(dummy_payload)
print(f"Model Prediction: {response}")

# 4. Cleanup (Always delete endpoints when finished to avoid bleeding money!)
predictor.delete_endpoint()
```

## Additional Resources

- [Amazon SageMaker Real-Time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [Auto-Scaling SageMaker Endpoint Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
