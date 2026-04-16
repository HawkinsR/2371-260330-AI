"""
Demo: SageMaker JumpStart Live Deployment & Inference
This script demonstrates how to programmatically deploy a pretrained 
JumpStart model and perform real-time inference.
"""

import sagemaker
import time
from sagemaker.jumpstart.model import JumpStartModel

def run_live_jumpstart_demo():
    print("="*60)
    print("  SageMaker JumpStart: Live Deployment Demo")
    print("="*60)
    
    # 1. Setup Session and Identify the Model
    # We'll use a lightweight BERT model for text classification
    model_id = "huggingface-tc-bert-base-uncased"
    model_version = "*" 
    
    try:
        # Get the real execution role if running in SageMaker
        role = sagemaker.get_execution_role()
    except ValueError:
        # Fallback for local testing (will fail on actual .deploy() call)
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        print("Note: Running outside SageMaker. Deployment will be skipped if credentials missing.")

    print(f"\nRole: {role}")
    print(f"\nModel ID: {model_id}")
    print(f"\nModel Version: {model_version}")


    # 2. Configure the JumpStart Model
    print(f"\n[1/3] Configuring Model: {model_id}...")
    model = JumpStartModel(
        model_id=model_id,
        model_version=model_version,
        role=role
    )

    predictor = None
    try:
        # 3. Deploy the Model
        # This provisions a real-time HTTPS endpoint on an EC2 instance.
        print(f"\n[2/3] Deploying to Real-time Endpoint (ml.g4dn.xlarge)...")
        print("      Estimated wait time: 3-6 minutes.")
        
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge"
        )
        
        print("\n✅ Endpoint is InService!")

        # 4. Perform Live Inference
        print(f"\n[3/3] Running Live Inference Test...")
        test_inputs = [
            "SageMaker JumpStart makes it incredibly easy to deploy foundation models.",
            "Well this is really cool!",
            "I am not happy abou this."
        ]

        for i in range(len(test_inputs)):
            start_time = time.time()
            response = predictor.predict(test_inputs[i])
            latency = time.time() - start_time

            print(f"\nResponse from Model:\n{response}")
            print(f"\nInference Latency: {latency:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Demo Failed: {e}")
        
    finally:
        # 5. Cleanup (CRITICAL)
        if predictor:
            print(f"\n[Cleanup] Deleting endpoint to stop billing...")
            predictor.delete_endpoint()
            print("✅ Endpoint successfully removed.")
        else:
            print("\n[Cleanup] No endpoint was created; no cleanup necessary.")
            
    print("\n" + "="*60)
    print("  Demo Complete")
    print("="*60)

if __name__ == "__main__":
    run_live_jumpstart_demo()
    
# Instructor Note:
# To show trainees how to use this in a separate notebook, they only need:
# from sagemaker.predictor import Predictor
# predictor = Predictor(endpoint_name='YOUR_ENDPOINT_NAME_HERE')
# response = predictor.predict({"instances": ["your text here"]})
