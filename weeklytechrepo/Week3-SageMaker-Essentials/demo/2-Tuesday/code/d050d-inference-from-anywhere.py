"""
SageMaker Essentials Demo (d050d) - Inference from Anywhere
Goal: Demonstrate how to communicate with a deployed model endpoint from a separate script.

PREREQUISITES:
1. You must have already run 'd050c-local-model-upload.py'.
2. You need the 'Endpoint Name' printed at the end of that script.
"""

import boto3
import json
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ==========================================================
# CONFIGURATION
# ==========================================================

# PASTE YOUR ENDPOINT NAME HERE (from the output of d050c)
ENDPOINT_NAME = "arn:aws:iam::407975137156:role/2371-SM-Execution-Test"

# ==========================================================
# METHOD 1: HIGH-LEVEL (SAGEMAKER SDK)
# Best for: Data Scientists working in SageMaker Studio/Notebooks
# ==========================================================
def predict_with_sdk(endpoint_name, data):
    print(f"\n--- Method 1: SageMaker SDK ---")
    try:
        # 1. 'Attach' to the existing endpoint
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # 2. Predict just like before
        print(f"Sending input: {data}")
        response = predictor.predict(data)
        print(f"Result: {response}")
        return response
    except Exception as e:
        print(f"SDK Error: {e}")

# ==========================================================
# METHOD 2: LOW-LEVEL (BOTO3)
# Best for: Software Engineers, AWS Lambda, External Web Apps
# ==========================================================
def predict_with_boto3(endpoint_name, data):
    print(f"\n--- Method 2: Boto3 (Runtime Client) ---")
    try:
        # 1. Create a runtime client
        runtime = boto3.client('sagemaker-runtime')
        
        # 2. Manually serialize the data to JSON
        payload = json.dumps(data)
        
        # 3. Invoke the endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # 4. Manually deserialize the response body
        result_body = response['Body'].read().decode('utf-8')
        result_data = json.loads(result_body)
        
        print(f"Sending input: {data}")
        print(f"Result: {result_data}")
        return result_data
    except Exception as e:
        print(f"Boto3 Error: {e}")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    if ENDPOINT_NAME == "PASTE_YOUR_ENDPOINT_NAME_HERE":
        print("\n" + "!"*50)
        print(" ACTION REQUIRED ".center(50, "!"))
        print("!"*50)
        print("Please edit this script and replace ENDPOINT_NAME with the")
        print("actual name from your 'd050c' deployment.")
        print("You can also find it in the AWS Console under:")
        print("SageMaker -> Inference -> Endpoints")
    else:
        test_data = [42.0, 15.5, -3.14]
        
        predict_with_sdk(ENDPOINT_NAME, test_data)
        predict_with_boto3(ENDPOINT_NAME, test_data)

        print("\n" + "="*50)
        print(" DEMO COMPLETE ".center(50, "="))
        print("="*50)
        print("Key Takeaways:")
        print("1. Predictor.attach() lets you reuse endpoints in different scripts.")
        print("2. 'sagemaker-runtime' is the 'lightweight' way to query models.")
        print("3. Always match the ContentType to your model's input_fn logic.")
