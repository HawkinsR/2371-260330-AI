# Lab: Automating the ML Lifecycle

## The Scenario
One-off training jobs are difficult to track and reproduce. Your organization is adopting **SageMaker Pipelines** to automate the end-to-end ML lifecycle. Your task is to define a **Directed Acyclic Graph (DAG)** that orchestrates the data processing, model training, and model registration steps into a single, repeatable workflow.

## Core Tasks

1. **Pipeline Definition (DAG):**
   - Open `e037-automating-the-ml-lifecycle.py`.
   - Instantiate a `MockSageMakerPipeline` named `"Enterprise-Auto-ML"`.
   - Add the following steps to the DAG:
     - **Step 1: Data-Preprint** (Clean raw S3 data).
     - **Step 2: Model-Train** (Execute the PyTorch Estimator).
     - **Step 3: Model-Register** (Submit to the Model Registry).
2. **Model Registry Integration:**
   - Within the "Model-Register" step, ensure the model is submitted to the `CustomerChurnPredictors` group.
   - Set the initial status to `PendingManualApproval`.
3. **Automated Execution:**
   - Call the `.execute()` method on your pipeline to simulate the sequential running of the DAG steps.
4. **Manual Approval Simulation:**
   - Simulate a Lead Data Scientist reviewing the pipeline's output and updating the Model Package status to `Approved`.

## Definition of Done
- A functional Python script that defines and "executes" a 3-step SageMaker Pipeline.
- The console output clearly shows the sequential transition from Processing -> Training -> Registration.
- The final model artifact is successfully moved to `Approved` status in the Registry.
