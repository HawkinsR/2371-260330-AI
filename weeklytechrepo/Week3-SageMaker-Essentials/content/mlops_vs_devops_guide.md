# MLOps: Evolution of the CI/CD Pipeline

For an engineer with a background in traditional DevOps (CI/CD and Git), MLOps feels familiar but introduces a layer of **complexity and non-determinism** that traditional software doesn't have.

## The Mental Shift: Code vs. Models

In traditional DevOps, your source code is the single source of truth. In MLOps, the truth is split across three dimensions.

| Feature | Traditional DevOps | MLOps (Machine Learning Operations) |
| :--- | :--- | :--- |
| **Primary Artifact** | Compiled Binary / Container | Trained Model Artifact (`model.tar.gz`) |
| **Source of Truth** | Source Code (Git) | **Code + Data + Hyperparameters** |
| **Pipeline Trigger** | Code Commit | Code Commit **OR** Data Change **OR** Schedule |
| **Testing** | Unit Tests, Integration (Deterministic) | Model Evaluation, Validation (Probabilistic) |
| **Drift** | Software doesn't "rot" (unless OS changes) | Models "decay" as the real world changes (**Data Drift**) |

---

## SageMaker's Four Pillars of Version Control

SageMaker doesn't use a single "Git repo" for everything. Instead, it versions the four main components of an ML system independently.

### 1. Code Versioning (Git Integration)
SageMaker Studio and Training Jobs integrate directly with standard Git providers (GitHub, GitLab, CodeCommit). 
- **The Best Practice:** Always attach a specific Git commit ID to your training jobs so you know exactly which version of `train.py` created a model.

### 2. Data Versioning (S3 & Feature Store)
Data is too large for Git. 
- **S3 Object Versioning:** Keeps historical copies of yours `.csv` or `.parquet` files.
- **SageMaker Feature Store:** Provides a "Point-in-Time" lookup. You can ask for "the features as they looked on January 1st," which is critical for reproducing a model.

### 3. Environment Versioning (Amazon ECR)
Machine learning depends heavily on specific library versions (PyTorch, CUDA, Scikit-Learn).
- SageMaker uses **Docker images** stored in Amazon ECR.
- By using immutable image tags (e.g., `v1.2.0` instead of `latest`), you ensure that a model built today will still work in a year.

### 4. Model Versioning (Model Registry)
This is the most critical pillar. The **Model Registry** is a catalog of your trained `model.tar.gz` files.
- You don't just "push" a model to production.
- You register a **Model Version** into a **Model Group**.
- It stays in `PendingManualApproval` until a human or an automated test suite marks it as `Approved`.

---

## The "CT" in CI/CD/CT

You are likely used to **Continuous Integration (CI)** and **Continuous Deployment (CD)**. MLOps adds **Continuous Training (CT)**.

- **CI:** Testing code syntax and packaging.
- **CD:** Moving the artifact to staging/production.
- **CT:** Detecting when model accuracy drops (Drift) and automatically triggering a **SageMaker Pipeline** to re-train the model on newer data.

## Visualizing the DAG (Directed Acyclic Graph)

Traditional CI/CD is often a linear sequence: `Build -> Test -> Deploy`.
MLOps uses a **DAG** because the dependencies are more complex. For example:
- You might have two different Preprocessing steps running in parallel.
- You might have a "Condition Step" that only registers the model if accuracy is > 90%.

> [!IMPORTANT]
> A SageMaker Pipeline is a managed DAG. It ensures that Step B (Training) never starts until Step A (Preprocessing) has successfully finished and uploaded its data to S3.

---

## Summary for the DevOps Engineer

- **Git is for Code**, but **Model Registry is for Models**.
- **Unit testing is for Logic**, but **Evaluation is for Quality**.
- **Jenkins/GitLab is for Software**, but **SageMaker Pipelines is for Workflows**.
