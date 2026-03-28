from unittest.mock import MagicMock
import json

# =====================================================================
# MOCK AWS INFRASTRUCTURE (Do not edit this section)
# =====================================================================
class MockModelRegistry:
    def __init__(self):
        self.packages = {}
        
    def create_model_package(self, group_name, metrics, approval_status, description, inference_spec):
        print(f"\n[Model Registry] Creating new model package in group: '{group_name}'")
        print(f"[Model Registry] Description: {description}")
        print(f"[Model Registry] Initial Status set to: {approval_status}")
        
        version = len(self.packages.get(group_name, [])) + 1
        package_arn = f"arn:aws:sagemaker:us-east-1:123456789:model-package/{group_name}/{version}"
        
        if group_name not in self.packages:
            self.packages[group_name] = []
        self.packages[group_name].append({
            "arn": package_arn,
            "status": approval_status
        })
        
        print(f"[Model Registry] ✅ Package successfully registered! ARN: {package_arn}")
        return package_arn

    def update_model_package(self, package_arn, new_status):
        print(f"\n[Model Registry] Updating Package Status for {package_arn}...")
        print(f"[Model Registry] Transitioning status to: {new_status}")
        print(f"[Model Registry] ✅ Status updated. Downstream pipelines may deploy this artifact.")

# =====================================================================
# YOUR TASKS
# =====================================================================
def register_new_model(registry):
    print("--- 1. Registering New Trained Artifact ---")
    
    # 1. Call registry.create_model_package()
    package_arn = registry.create_model_package(
        group_name="CustomerChurnPredictors",
        metrics={"ModelQuality": {"S3Uri": "s3://eval-bucket/accuracy.json"}},
        approval_status="PendingManualApproval",
        description="Completed training run ready for evaluation.",
        inference_spec={
            "Containers": [{"ModelDataUrl": "s3://artifact-bucket/model.tar.gz"}],
            "SupportedContentTypes": ["application/json"]
        }
    )
    
    return package_arn

def approve_model(registry, package_arn):
    print("\n--- 2. Simulating Human-In-The-Loop Approval ---")
    
    # 2. Call registry.update_model_package()
    registry.update_model_package(
        package_arn=package_arn,
        new_status="Approved"
    )

if __name__ == "__main__":
    aws_registry = MockModelRegistry()
    
    # Execution
    arn = register_new_model(aws_registry)
    
    if arn:
        approve_model(aws_registry, arn)
    else:
        print("ERROR: package_arn returning None. Finish the register_new_model function.")
