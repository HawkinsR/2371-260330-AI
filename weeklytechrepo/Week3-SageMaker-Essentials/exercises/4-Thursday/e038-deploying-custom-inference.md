# Lab: Deploying Custom Inference

## The Scenario
Your model is trained and registered, but it's not yet serving predictions. Your task is to write a custom **Inference Script** (`inference.py`) that SageMaker will use to load your model into a **Real-time Endpoint**. You must handle the deserialization of incoming JSON data, execute the model's forward pass, and format the output for the client.

## Core Tasks

1. **Inference Script Hooks:**
   - Open `e038-deploying-custom-inference.py`.
   - Implement the `model_fn(model_dir)` hook:
     - Determine the device (`cuda` if available, else `cpu`).
     - Instantiate an empty `RetailForecaster` model.
     - Construct the path to `model.pth` using `os.path.join(model_dir, 'model.pth')`.
     - If the file exists, load its weights: `model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))`. *(In this lab the file is simulated, so guard the call with `if os.path.exists(model_path)`.)*
     - Move the model to the device and set it to `.eval()` mode.

   - Implement the `predict_fn(input_data, model)` hook:
     - Move the input tensor to the correct device (CPU/GPU).
     - Execute the model within a `torch.no_grad()` context to optimize performance.
2. **Endpoint Deployment Simulation:**
   - Use the SageMaker Python SDK (simulated) to create a `Model` object pointing to your S3 artifact and your inference script.
   - Call the `.deploy()` method, specifying:
     - `initial_instance_count=1`
     - `instance_type='ml.t2.medium'` (CPU for cost-effective inference).
3. **Internal Testing:**
   - Run a test prediction against your simulated endpoint using a JSON payload.
   - Verify that the response matches the expected forecast format.

## Definition of Done
- A complete `inference.py` script with functional `model_fn` and `predict_fn` hooks.
- A simulated deployment sequence that successfully instantiates a Real-time Endpoint.
- Validated JSON output from a test prediction call.
