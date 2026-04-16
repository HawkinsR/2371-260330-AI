# Demo: MLOps and Model Registry

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **CI/CD Pipeline** | *"Currently, you manually click 'Run Training' whenever you have new data. What are the risks of that approach at enterprise scale? What would an automated pipeline watch for to know when to retrain?"* |
| **Model Drift** | *"A fraud detection model trained in 2020 on pre-pandemic data. How might fraudsters' behavior have changed by 2025? What metric would you monitor to detect that the model is getting worse?"* |
| **Model Registry** | *"What problems does Git solve for code (version history, rollback, collaboration)? Now imagine a model registry doing the same thing for trained ML artifacts. What extra information would each 'commit' need to store?"* |
| **Foundation Model / Prompt Engineering** | *"If AWS Bedrock gives you access to Claude or Llama via one API call, what situations would you still choose to train a custom model from scratch? What are the tradeoffs?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/mlops-pipeline.mermaid`.
2. Trace the path from Raw Data triggering a Processing Job, cascading into a Training Job, and finally an Evaluation Job. Explain that this entire flow is an automated "SageMaker Pipeline" Directed Acyclic Graph (DAG).
3. Walk through the **Model Registry** subgraph. 
4. **Discussion:** Ask the class: "Why does the pipeline stop at `PendingManualApproval` instead of deploying straight to Production?" (Answer: Machine learning models can degrade silently or harbor edge-case biases. High-stakes enterprise deployments require a "Human-in-the-Loop" to review the mathematical metrics before the code alters live user experiences).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d051-mlops-pipeline-packaging.py`.
2. Emphasize that we are using Mock classes to simulate the AWS `boto3` and `sagemaker` SDK calls safely.
3. Walk through `demonstrate_mlops_and_bedrock()`.
   - Point out the `group_name` parameter. Explain that over 5 years, this group acts like a GitHub repository tracking versions 1.0 through 99.0 of the specific Churn Predictor model.
   - Show how the inference specification literally points back to the `model.tar.gz` artifact generated in Wednesday's lesson.
4. Transition to the **AWS Bedrock Orientation**.
   - Explain that Bedrock flips the paradigm: instead of training a model for weeks, you simply pay per API call to utilize Foundation Models (like Claude) hosted completely serverlessly by AWS.
5. Execute the script. 
6. Show the sequence of logs: Registration -> Approval -> Bedrock Invocation.

## Summary
Reiterate that MLOps protects the Data Scientist from 3 AM pager alerts. By centralizing models in a Registry and automating drift detection, ML transitions from an isolated notebook process into a robust software engineering discipline.
