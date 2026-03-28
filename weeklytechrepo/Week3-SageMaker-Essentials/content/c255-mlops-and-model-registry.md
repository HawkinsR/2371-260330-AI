# MLOps and Model Registry

## Learning Objectives

- Introduce overarching MLOps principles comparing software engineering with ML engineering.
- Design CI/CD pipelines migrating static scripts into scheduled ML workflows.
- Monitor models in the wild establishing Model Decay and Drift concepts visually.
- Catalog active models within the SageMaker Model Registry utilizing versioning.
- Implement transparent Approval workflows signaling model readiness to external systems.
- Complete an AWS Bedrock Orientation surveying foundation models and prompt engineering.

## Why This Matters

A jupyter notebook is not production software. MLOps is the discipline of treating the machine learning lifecycle like a mature, version-controlled software repository. Furthermore, models degrade over time because the real world changes. If a model was trained on data from 2019, it will fail on inputs from 2025. You must continuously monitor for drift, trigger automated retraining pipelines, and log the distinct versions via a centralized registry. MLOps ensures ML is a reliable engineering discipline rather than a one-off science experiment.

## The Concept

### CI/CD and MLOps Pipelines

Standard DevOps pipelines trigger tests and deployments when you commit code. MLOps pipelines trigger when you commit code *OR* when new raw data arrives in an S3 bucket *OR* when the model's accuracy begins decaying below a threshold. Instead of manually clicking "run" on an Estimator, you link preprocessing, training, evaluation, and registration into a directed acyclic graph (using SageMaker Pipelines).

> **Key Term - CI/CD Pipeline (Continuous Integration / Continuous Deployment):** An automated workflow that triggers whenever code or data changes. CI automatically runs tests on every commit. CD automatically deploys passing builds to staging or production. In MLOps, the pipeline might trigger on a new dataset arriving in S3, automatically retraining, evaluating, and deploying the updated model.

### Data Drift vs Concept Drift

- **Data Drift:** The statistical distribution of the input data changes over time. (e.g., a camera lens gets scratched, changing the pixel inputs).
- **Concept Drift:** The relationship between the input and the target label changes over time. (e.g., consumer behavior shifts dramatically post-pandemic).
When drift is detected, pipelines automatically retrain the model on fresh historical data.

> **Key Term - Model Decay / Drift:** The gradual degradation of a deployed model's performance over time because the real world has changed in ways the model was never trained to handle. A model trained on 2019 customer data will produce increasingly inaccurate predictions as customer behavior evolves toward 2025. Drift is detected by comparing live prediction distributions against historical baselines.

### SageMaker Model Registry and Approval

The Model Registry is the "Git" for trained artifacts. It tracks every version of your model alongside the metrics (accuracy, F1) that generated it. A Model Package can have a status of "PendingManualApproval," "Approved," or "Rejected." Real-world deployment scripts should be configured to automatically pull down and deploy whichever package was most recently marked "Approved" by a Lead Data Scientist.

### AWS Bedrock Orientation

While SageMaker handles custom PyTorch training, AWS Bedrock provides a managed, serverless interface to query massive Foundation Models (like Anthropic's Claude or Meta's Llama) via API. If you do not need to train your own custom weights from scratch, Bedrock provides instant AI capability requiring nothing but MLOps prompt engineering.

> **Key Term - Foundation Model:** A large pre-trained model (like GPT-4 or Claude) trained by a major AI lab on enormous datasets at immense cost. Rather than training their own model from scratch, most organizations access Foundation Models via API and customize them through prompt engineering or fine-tuning for their specific use case.

> **Key Term - Prompt Engineering:** The practice of crafting the text input ("prompt") sent to a language model to guide it toward producing specific, useful outputs. Effective prompts include clear instructions, relevant context, output format requirements, and examples — enabling non-trivial AI behaviors without any model retraining.

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
