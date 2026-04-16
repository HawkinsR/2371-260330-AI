import sagemaker
from sagemaker.pytorch import PyTorch
# Note: e036-train.py is the remote script that SageMaker will execute on the cluster.

def launch_estimator():
    """
    Task 2 & 4: Configure the SageMaker Estimator and metric definitions.
    """
    dummy_role = "arn:aws:iam::123456789012:role/FakeSageMakerRole"
    print(f"Using IAM Role: {dummy_role}")

    # TODO (Task 4): Define metric_definitions to capture training loss from stdout.
    # The e036-train.py script prints lines like: "Epoch 1 - Loss: 0.5000"
    # Use a regex pattern to extract the float value after "Loss: "
    metric_definitions = [
        # TODO: Uncomment and complete the entry below:
        # {"Name": "train:loss", "Regex": "Loss: ([0-9\\.]+)"}
    ]

    # TODO (Task 2): Configure the PyTorch Estimator
    # entry_point='e036-train.py'
    # instance_type='ml.g4dn.xlarge'
    # hyperparameters={'epochs': 50, 'lr': 0.001}
    # metric_definitions=metric_definitions
    estimator = None

    return estimator
def validate_prompt_input(user_input: str) -> str:
    """
    Task 5: Input Validation and Sanitation.
    Protects the system from malformed or malicious prompt inputs.
    """
    # TODO 1: Raise ValueError if user_input is an empty string (after stripping whitespace)

    # TODO 2: Raise ValueError if len(user_input) exceeds 500 characters

    # TODO 3: Raise ValueError if the phrase "ignore all previous instructions"
    #          appears in user_input (case-insensitive check)

    # If all checks pass, return the stripped input
    return user_input.strip()


if __name__ == "__main__":
    print("--- Local Launcher Script ---")

    # ReAct Prompt (Task 1) — document your reasoning here as comments:
    # Thought: I need to train a ResNet model for 50 epochs on a GPU instance.
    # Action: LaunchEstimator[instance=ml.g4dn.xlarge, epochs=50, lr=0.001]
    # Observation: (Will be filled in after the simulated .fit() completes)

    # Task 5: Validate the prompt input before launching
    test_prompt = "Train a ResNet18 model for bird classification."
    try:
        clean_prompt = validate_prompt_input(test_prompt)
        print(f"Validated prompt: '{clean_prompt}'")
    except ValueError as e:
        print(f"Input rejected: {e}")

    ESTIMATOR = launch_estimator()

    if ESTIMATOR:
        print("\nConfiguring .fit() to launch AWS instances...")
        # TODO: Call ESTIMATOR.fit() passing the S3 dictionary pointing to
        # 's3://my-enterprise-data-bucket/v1/' under the 'train' key.

        # This line runs e036-train.py locally to simulate the remote container
        import subprocess, sys
        subprocess.run([sys.executable, "e036-train.py"])
    else:
        print("Estimator is None — complete the TODO in launch_estimator().")

