# Weekly Epic: Modernize workflows by taking custom PyTorch models to the cloud using SageMaker ecosystems for training and production deployments

## 1-Monday

### Written Content

- [ ] Create `c251-sagemaker-ecosystem-and-setup.md`: SageMaker Ecosystem & Lifecycle, IAM best practices, Studio Domains & Profiles, JumpStart Pre-built Models.
- [ ] Create `c251b-prompting-paradigms-part-1.md`: Zero-Shot & Few-Shot Prompting, Chain of Thought & Tree of Thought, Graph of Thought & Dialog State, Curating Responses & Prompt Design.

### Instructor Demo

- [ ] Create `d049-sagemaker-studio-and-jumpstart.py`: Configure a SageMaker Studio domain and demonstrate deploying a pre-built model via JumpStart.

### Trainee Exercise

- [ ] Create `e035-exploring-sagemaker-studio.md`: Set up an IAM execution role, create a Studio profile, and perform basic prompt engineering tests in a JumpStart notebook.

## 2-Tuesday

### Written Content

- [ ] Create `c252-advanced-prompting-and-script-mode.md`: Input Validation and Sanitation, Chain of Verification & ReAct Prompting, Preventing Hallucinations & Parameterization, BYOM/BYOS & Script Mode Patterns, Estimators & Configurations, Launching Training Jobs.

### Instructor Demo

- [ ] Create `d050-advanced-prompting-and-estimations.py`: Prepare a ReAct-style prompt and launch a SageMaker training job using the PyTorch Estimator SDK.

### Trainee Exercise

- [ ] Create `e036-re-act-and-estimators.md`: Implement a ReAct prompt sequence and successfully launch a custom training run from a local script.

## 3-Wednesday

### Written Content

- [ ] Create `c253-lifecycle-and-mlops-pipelines.md`: Model Artifacts & S3 Storage, `model.tar.gz` Structure & Packaging, Pipelines, DAGs & Versioning, MLOps & CI/CD Principles, Introduction to Model Registry.

### Instructor Demo

- [ ] Create `d051-mlops-pipeline-packaging.py`: Package a model artifact and register it in a SageMaker Pipeline or Model Registry.

### Trainee Exercise

- [ ] Create `e037-automating-the-ml-lifecycle.md`: Create a basic SageMaker Pipeline to automate step-wise execution from packaging to registration.

## 4-Thursday

### Written Content

- [ ] Create `c254-deployment-and-inference-scripts.md`: Real-time Inference Endpoints, Instance Selection, `model_fn` & `predict_fn` Logic, Auto-scaling Configuration, Approval Workflows & Versioning.

### Instructor Demo

- [ ] Create `d052-serving-and-inference.py`: Write an `inference.py` and deploy it to a real-time endpoint, testing request/response cycles.

### Trainee Exercise

- [ ] Create `e038-deploying-custom-inference.md`: Deploy a persistent endpoint with custom `inference.py` logic, test latency, and tear down securely.

## 5-Friday

### Written Content

- [ ] Create `c255-admin-bedrock-and-costs.md`: AWS Bedrock Orientation, Runtime API & Inference Parameters, API Quotas & Handling Throttling, CloudWatch Monitoring & Guardrails, CLI/Boto3 SDK Setup, Cost Management & Policies.

### Instructor Demo

- [ ] Create `d053-boto3-bedrock-and-costs.py`: Access Bedrock via Boto3 and demonstrate cost-tracking tags or CloudWatch metric pulls.

### Trainee Exercise

- [ ] Create `e039-enterprise-admin-tasks.md`: Configure Boto3 credentials, monitor endpoint usage in CloudWatch, and analyze cost projection via SageMaker Cost Explorer.

## Project 3: SecureContent AI

- [ ] Create `p3-secure-content-ai.md`: Multi-modal audit pipeline combining custom PyTorch CNNs with ReAct prompting and FastAPI integration.
