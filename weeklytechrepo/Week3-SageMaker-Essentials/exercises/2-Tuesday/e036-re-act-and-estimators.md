# Lab: ReAct Prompting and SageMaker Estimators

## The Scenario

Your model training is now handled by SageMaker, but you need an intelligent interface to manage these jobs. Your task is to implement a **ReAct (Reason + Act)** prompting loop that can "think" about a training requirement, "act" by configuring a SageMaker **Estimator**, and then "observe" the resulting CloudWatch metrics to decide if the job was successful.

## Core Tasks


1. **ReAct Prompt Design:**
   - Define a multi-step prompt that includes:
     - **Thought:** "I need to train a ResNet model for 50 epochs on a GPU instance."
     - **Action:** `LaunchEstimator[instance=ml.g4dn.xlarge, epochs=50]`
     - **Observation:** (Simulated) "Training Loss: 0.12, Validation Accuracy: 94%"
2. **Estimator Configuration (Script Mode):**
   - Open `e036-re-act-and-estimators.py`.
   - Configure a `PyTorch` Estimator with:
     - `entry_point='e036-train.py'`
     - `instance_type='ml.g4dn.xlarge'`
     - `hyperparameters={'epochs': 50, 'lr': 0.001}`
3. **Hyperparameter Logic:**
   - Ensure the `e036-train.py` script correctly uses `argparse` to read the `epochs` and `learning_rate` passed from the ReAct loop.
4. **Metric Tracking:**
   - Define a `metric_definitions` list to capture `'train:loss'` from stdout and map it to SageMaker metrics. *Hint: The `e036-train.py` script prints lines in the format `"Epoch N - Loss: X.XXXX"`. Your regex should capture the float after `"Loss: "`.*
5. **Input Validation:**
   - Implement the `validate_prompt_input` function in the launcher script.
   - It should raise a `ValueError` for empty strings, inputs longer than 500 characters, and inputs containing the phrase `"ignore all previous instructions"`.

## Definition of Done

- A documented ReAct prompt sequence that successfully reasons through a training requirement.
- A functional `e036-re-act-and-estimators.py` script that instantiates a PyTorch Estimator.
- Successful simulation of a `.fit()` call pointing to an S3 data channel.
- The `metric_definitions` list is populated with a working regex pattern for `train:loss`.
- The `validate_prompt_input` function correctly raises `ValueError` for the three edge cases.
