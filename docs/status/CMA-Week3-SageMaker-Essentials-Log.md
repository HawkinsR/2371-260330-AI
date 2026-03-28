# Weekly Epic: Modernize workflows by taking custom PyTorch models to the cloud using SageMaker ecosystems for training and production deployments

## 1-Monday

### Written Content

- [x] Create `c251-sagemaker-ecosystem-and-script-mode.md`: SageMaker Ecosystem & Lifecycle, IAM best practices, BYOM/BYOS & Script Mode Patterns, Script Mode Interface, Connecting Local Code to Cloud, Estimator vs Processor Concepts.

### Instructor Demo

- [x] Create `d049-sagemaker-script-mode-setup.py`: Prepare a local entry point python script for SageMaker script mode and configure the associated AWS IAM roles.

### Trainee Exercise

- [x] Create `e035-local-to-cloud-script.md`: Convert an existing PyTorch script into an execution-ready Script Mode file passing appropriate system variables.

## 2-Tuesday

### Written Content

- [x] Create `c252-sagemaker-estimators-and-training.md`: SageMaker Estimators, Adapting PyTorch Scripts for Cloud, The `Estimator` Class & SDK, Passing Hyperparameters & Metrics, Handling Dependencies (`requirements.txt`), Launching Training Jobs.

### Instructor Demo

- [x] Create `d050-launching-custom-estimators.py`: Trigger a remote asynchronous training job using the PyTorch Estimator SDK, passing hyper-parameters and dependencies.

### Trainee Exercise

- [x] Create `e036-sagemaker-training-job.md`: Configure dependency injection requirements, map metric definitions, and successfully launch the SageMaker training run.

## 3-Wednesday

### Written Content

- [x] Create `c253-model-packaging-and-inference.md`: Model Artifacts & Inference, `model.tar.gz` Structure & Packaging, Creating `inference.py` Entry Point, `model_fn` (Loading Logic), `predict_fn` (Inference Logic), Testing Inference Capabilities.

### Instructor Demo

- [x] Create `d051-custom-inference-script.py`: Package the `state_dict` into a `model.tar.gz` and write `model_fn` and `predict_fn` inference scripts.

### Trainee Exercise

- [x] Create `e037-deploying-model-tarball.md`: Package a trained artifact and write a reliable `inference.py` with custom parsing logic prior to cloud tests.

## 4-Thursday

### Written Content

- [x] Create `c254-sagemaker-endpoints-and-scaling.md`: Deployment Essentials, Real-time Endpoints architecture, Instance selection: CPU vs GPU vs Inferentia, Auto-scaling configuration, Updating & Rolling Back Endpoints, Cost Optimization Basics.

### Instructor Demo

- [x] Create `d052-endpoint-deployment-and-update.py`: Deploy a real-time endpoint based on the trained artifact, invoking the endpoint for payload inference.

### Trainee Exercise

- [x] Create `e038-managing-realtime-endpoints.md`: Launch a persistent endpoint with auto-scaling bounds, test real-time latencies, and tear down the endpoint securely.

## 5-Friday

### Written Content

- [x] Create `c255-mlops-and-model-registry.md`: Production Basics, Introduction to MLOps principles, CI/CD for ML pipelines, Model decay and drift concepts, SageMaker Model Registry, Approval workflows, Moving from Notebook to Pipeline, AWS Bedrock Orientation (Foundation Models).

### Instructor Demo

- [x] Create `d053-sagemaker-mlops-pipeline.py`: Showcase registering a newly trained model within the SageMaker Model Registry, checking status and basic Bedrock access.

### Trainee Exercise

- [x] Create `e039-model-registry-workflow.md`: Version control an active model into the SageMaker Registry, transition it to Approved status, and explore basic MLOps drift logic.
