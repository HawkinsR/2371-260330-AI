# Lab: Local to Cloud Script Mode

## The Scenario
Your company currently trains its PyTorch models by manually running a python script (`train_model.py`) on a local laptop. The dataset is getting too large, and training now takes 3 days. Your manager wants to move this workload to AWS SageMaker using "Script Mode." Your task is to refactor the hardcoded local script so that it can dynamically accept hyperparameters from SageMaker and locate training data injected by the cloud container.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e035-cloud_script_lab.py`.
3. Read the `train_model` function to understand what the code is doing (simulating a basic training loop).
4. Complete the `main` execution block:
   - Initialize an `argparse.ArgumentParser()`.
   - Add three standard hyperparameters that SageMaker will need to pass: `--epochs` (int, default 5), `--batch-size` (int, default 32), and `--learning-rate` (float, default 0.001).
   - Add the two essential SageMaker Environment Variables. Use `os.environ.get()` to set the defaults:
     - `--model-dir` defaults to `SM_MODEL_DIR` (or `./model` if not found).
     - `--train` defaults to `SM_CHANNEL_TRAIN` (or `./data` if not found).
   - Parse the arguments.
   - Pass the parsed arguments into the `train_model` function call.

## Definition of Done
- The script executes successfully locally.
- It prints: `Loading training data from: ./data`
- It prints: `Training with epochs: 5, batch_size: 32, lr: 0.0010`
- It prints: `Saving final model artifact to: ./model`
