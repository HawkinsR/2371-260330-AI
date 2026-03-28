# Lab: Model Registry Workflow

## The Scenario
Your CI/CD training pipeline has just successfully produced a `.tar.gz` model artifact that achieved 96% accuracy on the evaluation dataset. Rather than emailing this file to the DevOps team, your organization mandates that all models be strictly version-controlled within the SageMaker Model Registry. Your task is to register this new artifact version under the `CustomerChurnPredictors` Model Package Group, but assign it a `PendingManualApproval` status. Once registered, simulate the Lead Data Scientist reviewing the metrics and updating the package status to `Approved` so the deployment pipeline can pick it up.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e039-registry_lab.py`.
3. Complete the `register_new_model` function:
   - Call `registry.create_model_package()`.
   - Pass `group_name` as `"CustomerChurnPredictors"`.
   - Pass `metrics` as a dictionary pointing `"ModelQuality"` to `{"S3Uri": "s3://eval-bucket/accuracy.json"}`.
   - Set `approval_status` to `"PendingManualApproval"`.
   - Set `description` to something descriptive.
   - Set `inference_spec` to `{"Containers": [{"ModelDataUrl": "s3://artifact-bucket/model.tar.gz"}], "SupportedContentTypes": ["application/json"]}`.
   - Return the resulting `package_arn`.
4. Complete the `approve_model` function:
   - Call `registry.update_model_package()`.
   - Pass the provided `package_arn`.
   - Update the `new_status` to `"Approved"`.

## Definition of Done
- The script executes successfully locally using the provided Mock environment.
- The output confirms that a new model package was registered under the intended group.
- The console clearly shows the transition from `PendingManualApproval` to `Approved`.
