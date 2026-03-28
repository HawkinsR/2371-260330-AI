"""
Demo: SageMaker MLOps Pipeline and Registry
This script simulating the registration of a newly trained model into the 
SageMaker Model Registry, transitioning its approval status, and briefly 
touching upon AWS Bedrock access for Foundation Models.

Note: Execution is mocked to demonstrate SDK logic safely without real AWS billing.
"""

from unittest.mock import MagicMock
import json

class MockModelRegistry:
    """Simulates the SageMaker Model Registry service."""
    def __init__(self):
        self.packages = {}
        
    def create_model_package(self, group_name, metrics, approval_status, description, inference_spec):
        """Simulates registering a model artifact into the registry catalog."""
        print(f"\n[Model Registry] Creating new model package in group: '{group_name}'")
        print(f"[Model Registry] Description: {description}")
        print(f"[Model Registry] Attaching evaluation metrics (simulated JSON)...")
        print(f"[Model Registry] Initial Status set to: {approval_status}")
        
        # Determine version number based on existing packages in the group
        version = len(self.packages.get(group_name, [])) + 1
        # ARN (Amazon Resource Name) uniquely identifies the package globally in AWS
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
        """Simulates a manager or automated pipeline altering the approval status."""
        print(f"\n[Model Registry] Updating Package Status for {package_arn}...")
        print(f"[Model Registry] Transitioning status to: {new_status}")
        print(f"[Model Registry] ✅ Status updated. Downstream CI/CD pipelines may now deploy this artifact.")

class MockBedrockRuntime:
    """Simulates the AWS Bedrock client for interacting with hosted Foundation Models (like Claude/Llama)."""
    def invoke_model(self, body, modelId, accept, contentType):
        print(f"\n[AWS Bedrock] Invoking Foundation Model: {modelId}")
        print(f"[AWS Bedrock] Parsing prompt payload...")
        
        # Simulate an LLM text generation response
        response_body = {"completion": " MLOps is the practice of integrating machine learning into the software development lifecycle to ensure reliable, scalable, and automated deployment."}
        
        # Mocking the boto3 streaming response structure that AWS returns
        class StreamResponse:
            def read(self):
                return json.dumps(response_body).encode('utf-8')
                
        return {"body": StreamResponse()}

def demonstrate_mlops_and_bedrock():
    print("--- SageMaker MLOps and Model Registry ---")
    print("Scenario: A training pipeline just finished. We must log the artifact.")
    
    registry = MockModelRegistry()
    
    # 1. Register the Model (Simulating standard SageMaker SDK call)
    print("\n1. Registering the trained artifact...")
    # These parameters closely match the real `sagemaker.Session().create_model_package()` structure
    package_arn = registry.create_model_package(
        group_name="CustomerChurnPredictors", # Group acts as a "folder" for versions of the same model
        metrics={"ModelQuality": {"S3Uri": "s3://eval-bucket/accuracy.json"}}, # Attach the test metrics for auditors
        approval_status="PendingManualApproval", # Default status prevents automatic deployment
        description="ResNet18 trained on August 2025 dataset with 94% accuracy.",
        inference_spec={ # Tell AWS what image and weights to load when deploying this version
            "Containers": [{"ModelDataUrl": "s3://artifact-bucket/model.tar.gz"}],
            "SupportedContentTypes": ["application/json"]
        }
    )
    
    # 2. Approve the Model (Simulating a Lead Data Scientist reviewing metrics)
    print("\n2. Simulating Human-In-The-Loop Approval...")
    # Usually triggers an EventBridge event to start a deployment pipeline automatically
    registry.update_model_package(
        package_arn=package_arn,
        new_status="Approved" 
    )
    
    # 3. AWS Bedrock Orientation (Foundation Models)
    print("\n--- AWS Bedrock Orientation ---")
    print("Scenario: We want to augment our classification app with an LLM text summary without training a model.")
    
    bedrock = MockBedrockRuntime()
    # Claude models specifically expect prompt formats like "\n\nHuman: [text]\n\nAssistant:"
    prompt_data = {"prompt": "\n\nHuman: Briefly explain MLOps.\n\nAssistant:", "max_tokens_to_sample": 50}
    
    print("\n3. Invoking Anthropic Claude via Bedrock API...")
    # Invoke the model by paying per-token rather than renting an entire server
    response = bedrock.invoke_model(
        body=json.dumps(prompt_data),
        modelId="anthropic.claude-v2", # The ID of the model instance we want to query
        accept="application/json",
        contentType="application/json"
    )
    
    # Parse the raw stream response payload back into JSON
    response_body = json.loads(response.get('body').read())
    print(f"\n[Bedrock Output]:{response_body.get('completion')}")
    
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_mlops_and_bedrock()
