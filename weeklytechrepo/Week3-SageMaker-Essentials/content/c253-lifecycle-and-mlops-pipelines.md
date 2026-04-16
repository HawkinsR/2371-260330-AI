# Lifecycle and MLOps Pipelines

## Learning Objectives

- Understand Model Artifacts and S3 Storage lifecycle.
- Structure and package `model.tar.gz` for SageMaker compatibility.
- Design CI/CD pipelines migrating static scripts into scheduled ML workflows (DAGs).
- Catalog active models within the SageMaker Model Registry utilizing versioning.
- Implement transparent Approval workflows signaling model readiness to external systems.

## Why This Matters

A jupyter notebook is not production software. **MLOps** is the discipline of treating the machine learning lifecycle like a mature, version-controlled software repository. 

Building a model is only half the task; you must also package it correctly, store it securely in S3, and orchestrate the transition from training to deployment using automated pipelines. MLOps ensures that your ML workflows are reliable, repeatable, and audit-ready engineering processes rather than one-off science experiments.

## The Concept

### Model Artifacts and Packaging

When a SageMaker training job completes, it looks inside the `/opt/ml/model` directory of the container. Anything found there is automatically compressed into a single file named `model.tar.gz` and uploaded to your S3 bucket.

For a PyTorch model, this tarball typically contains:
- `model.pth`: The serialized weights (`state_dict`).
- `code/`: A directory containing your `inference.py` and `requirements.txt` for deployment.

> **Key Term - Model Artifact:** The physical file (typically a `.tar.gz`) that contains the trained weights and code needed to run a model. In SageMaker, this artifact is the output of an Estimator and the input of a Predictor.

> **Key Term - model.tar.gz:** The specific archive format required by SageMaker. It must contain the model weights at the root and optionally a `code/` folder for custom inference logic. If this structure is incorrect, the deployment will fail to load the model.

### SageMaker Model Registry and Approval

The Model Registry is the "Git" for trained artifacts. It tracks every version of your model alongside the metrics (accuracy, F1) that generated it. A Model Package can have a status of **PendingManualApproval**, **Approved**, or **Rejected**. Real-world deployment scripts should be configured to automatically pull down and deploy whichever package was most recently marked "Approved" by a Lead Data Scientist.

> **Key Term - Model Registry:** A centralized repository within SageMaker to catalog models for production. It allows team members to version models, associate metadata, and manage the approval lifecycle before deployment.

### CI/CD and MLOps Pipelines (DAGs)

Standard DevOps pipelines trigger tests and deployments when you commit code. **MLOps Pipelines** trigger when you commit code *OR* when new raw data arrives in an S3 bucket. Instead of manually clicking "run" on an Estimator, you link preprocessing, training, evaluation, and registration into a **Directed Acyclic Graph (DAG)** using SageMaker Pipelines.

> **Key Term - DAG (Directed Acyclic Graph):** A mathematical structure of nodes and edges where the edges have a direction and no paths exist that start and end at the same node. In MLOps, a DAG represents a workflow where one step (e.g., Training) must wait for the previous step (e.g., Preprocessing) to complete successfully.

### Defining a SageMaker Pipeline

A SageMaker Pipeline is constructed by creating individual `Step` objects and chaining them together. The SDK then compiles these steps into a DAG and submits it to AWS for orchestrated execution. Steps automatically pass their outputs (e.g., a processed dataset or a trained model artifact) as inputs to downstream steps.

```python
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
session = sagemaker.Session()

# --- Step 1: Preprocessing ---
# A processing job that cleans raw data and splits it into train/test
processor = SKLearnProcessor(
    framework_version="1.0-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role
)
step_process = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[sagemaker.processing.ProcessingInput(
        source="s3://my-bucket/raw-data/",
        destination="/opt/ml/processing/input"
    )],
    outputs=[sagemaker.processing.ProcessingOutput(
        output_name="train_data",
        source="/opt/ml/processing/output/train"
    )],
    code="preprocess.py"  # Your local preprocessing script
)

# --- Step 2: Training ---
# Reads the output of the Preprocessing step as its input channel
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.0",
    py_version="py310"
)
step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            # Feeds the output of step_process directly into the training job
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
        )
    }
)

# --- Assemble and run the pipeline ---
pipeline = Pipeline(
    name="MyMLPipeline",
    steps=[step_process, step_train],  # Order matters: DAG dependency is inferred
    sagemaker_session=session
)
pipeline.upsert(role_arn=role)  # Create or update the pipeline definition in AWS
pipeline.start()                # Trigger an execution run
```

> **Key Term - Pipeline Step:** A discrete, managed unit of work within a SageMaker Pipeline. Each step maps to an underlying SageMaker job (Processing, Training, Transform) and automatically handles passing outputs from one step as inputs to the next, forming the DAG.

### Versioning and Registry Metadata Best Practices

The Model Registry is only useful if you treat each version as a formal record. When registering a model package, attach as much metadata as possible:

- **Model metrics** (accuracy, F1, AUC) recorded at evaluation time — so reviewers can compare versions without re-running experiments.
- **Dataset lineage** — the S3 URI of the exact training dataset used. This is critical for reproducibility and regulatory compliance.
- **Tags** — environment tags (`{"env": "staging"}`), team ownership, and compliance tags (e.g., `{"PII-free": "true"}`).
- **Description** — a human-readable changelog noting what changed between this version and the previous one.

This metadata discipline means that when something breaks in production, you can trace exactly which data, code, and parameters produced the deployed model.

## Code Example

```python
import sagemaker
from sagemaker import ModelMetrics, MetricsSource

# 1. Provide S3 evaluation metrics detailing how the trained model performed
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="s3://my-bucket/training-runs/eval/accuracy.json",
        content_type="application/json"
    )
)

# 2. Register the trained artifact as a new version within a distinct Model Group
sagemaker.Session().create_model_package(
    model_package_group_name="Customer-Churn-Predictors",
    model_metrics=model_metrics,
    approval_status="PendingManualApproval", # Must be approved by a human to deploy
    description="ResNet18 trained on August 2025 dataset with 94% accuracy.",
    # We point to the underlying training artifact
    inference_specification={
        "Containers": [{"ModelDataUrl": "s3://my-bucket/artifacts/model.tar.gz",
                        "Image": "pytorch-inference:2.0.0"}],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"]
    }
)
print("Model Version successfully registered and awaiting approval.")
```

## Additional Resources

- [SageMaker MLOps Pipelines Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html)
- [SageMaker Model Registry Concept](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [SageMaker Pipelines SDK Reference](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html)
