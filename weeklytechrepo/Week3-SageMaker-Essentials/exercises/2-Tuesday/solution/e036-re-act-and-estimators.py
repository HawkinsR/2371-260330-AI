import sagemaker
from sagemaker.pytorch import PyTorch
# Note: e036-train.py is the remote script that SageMaker will execute on the cluster.

def launch_estimator():
    """
    Task 2 & 4: Configure the SageMaker Estimator and metric definitions.
    """
    dummy_role = "arn:aws:iam::123456789012:role/FakeSageMakerRole"
    print(f"Using IAM Role: {dummy_role}")

    # Task 4: Define metric_definitions to capture training loss from stdout.
    metric_definitions = [
        {"Name": "train:loss", "Regex": "Loss: ([0-9\\.]+)"}
    ]

    # Task 2: Configure the PyTorch Estimator
    estimator = PyTorch(
        entry_point='e036-train.py',
        source_dir='./', # Assuming we run from the correct directory locally for simulation
        role=dummy_role,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type='ml.g4dn.xlarge',
        hyperparameters={
            'epochs': 50,
            'learning_rate': 0.001
        },
        metric_definitions=metric_definitions
    )

    return estimator

def validate_prompt_input(user_input: str) -> str:
    """
    Task 5: Input Validation and Sanitation.
    Protects the system from malformed or malicious prompt inputs.
    """
    stripped_input = user_input.strip()
    
    # 1: Raise ValueError if user_input is an empty string
    if not stripped_input:
        raise ValueError("Input prompt cannot be empty.")

    # 2: Raise ValueError if len(user_input) exceeds 500 characters
    if len(stripped_input) > 500:
        raise ValueError("Input prompt exceeds max length of 500 characters.")

    # 3: Raise ValueError for prompt injection phrase
    if "ignore all previous instructions" in stripped_input.lower():
        raise ValueError("Potentially malicious prompt override detected.")

    return stripped_input


if __name__ == "__main__":
    print("--- Local Launcher Script ---")

    # ReAct Prompt (Task 1) — document your reasoning here as comments:
    # Thought: I need to train a ResNet model for 50 epochs on a GPU instance.
    # Action: LaunchEstimator[instance=ml.g4dn.xlarge, epochs=50, lr=0.001]
    # Observation: Local estimation simulated. AWS API will instantiate a container.

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
        try:
            # Simulate S3 integration
            ESTIMATOR.fit({'train': 's3://my-enterprise-data-bucket/v1/'})
        except:
            pass

        import subprocess, sys
        import os
        # Simulate local run
        train_script = os.path.join(os.path.dirname(__file__), "e036-train.py")
        if os.path.exists(train_script):
             subprocess.run([sys.executable, train_script])
        else:
            print(f"Notice: Ensure the solution folder contains 'e036-train.py' for local execution.")
    else:
        print("Estimator is None — complete the TODO in launch_estimator().")
