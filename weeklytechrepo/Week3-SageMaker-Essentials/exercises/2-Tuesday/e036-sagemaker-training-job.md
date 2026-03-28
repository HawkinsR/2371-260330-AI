# Lab: Launching a SageMaker Training Job

## The Scenario
You have successfully refactored your PyTorch script to accept SageMaker environment variables (`e036-train.py`). Now, you need to write the "launcher" script (`e036-launch_job.py`) that will actually command AWS to provision a GPU instance, copy your training data from S3, and execute your script. You also need to ensure that an external dependency (`pandas`) is installed on the cloud container before your script runs.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e036-requirements.txt` and add `pandas==2.1.0` to ensure the container installs it before running your script.
3. Open `e036-launch_job.py`.
4. Complete the `launch_estimator` function:
   - Instantiate a `PyTorch` Estimator.
   - Set the `entry_point` to point to `e036-train.py`.
   - Set the `source_dir` to point to the current directory (`.` or `./`) so that AWS uploads both `e036-train.py` and `e036-requirements.txt` together.
   - Set the `role` using the provided dummy ARN string.
   - Specify `framework_version='2.0.0'` and `py_version='py310'`.
   - Request `1` instance of type `ml.g4dn.xlarge` (a standard GPU instance).
   - Pass a `hyperparameters` dictionary containing: `'epochs': 15` and `'learning_rate': 0.002`.
   - Add a `metric_definitions` list containing a dictionary with Name: `'train:loss'` and Regex: `'Loss: ([0-9\\.]+)'` so SageMaker can map your print statements to CloudWatch graphs.
5. In the final `if __name__ == "__main__":` block, call the `.fit()` method on your returned estimator.
   - Pass a dictionary mapping the channel name `'train'` to a simulated S3 URI: `'s3://my-enterprise-data-bucket/v1/'`.

## Definition of Done
- `e036-requirements.txt` lists pandas.
- `e036-launch_job.py` configures the PyTorch Estimator with all requested parameters.
- Calling the script initiates the simulated remote training execution via the `.fit()` method.
