# Lab: Deploying Model Tarballs and Inference Scripts

## The Scenario
Your SageMaker Training Job has successfully completed. AWS has taken your final `model.pth` file and zipped it into a `model.tar.gz` artifact stored safely in S3. Now, you need to write the `inference.py` script that will tell the SageMaker prediction instances exactly how to load those weights back into memory and how to structure the inference logic when an HTTP request arrives.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e037-inference_lab.py`.
3. Read the `RetailForecaster` class. This is the simple architecture you must map your loaded weights onto.
4. Complete the `model_fn` function:
   - Determine the computational `device` (`cuda` if available, else `cpu`).
   - Instantiate an empty `RetailForecaster` architecture.
   - Construct the full path to `model.pth` using the provided `model_dir`.
   - Use `torch.load()` (make sure to use `map_location=device`), and pass that into `model.load_state_dict()`.
   - Call `model.to(device)` followed immediately by `model.eval()`.
5. Complete the `predict_fn` function:
   - Determine the `device` and send the `input_data` tensor to it.
   - Wrap your prediction call inside a `torch.no_grad()` context manager to save memory.
   - Pass the `input_data` through the `model` and return the `prediction` tensor.

## Definition of Done
- The script executes without errors.
- The `e037-inference_lab.py` simulates the endpoint lifecycle successfully.
- The final output prints a simulated JSON response verifying the weights successfully loaded and predictions successfully returned.
