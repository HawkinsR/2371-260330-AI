"""
Demo: Boto3 Bedrock and Costs (Live)
This script demonstrates how to interact with the AWS Bedrock Runtime API 
and AWS Cost Explorer using the Boto3 SDK.
"""

import boto3
import json
import time
import random
from datetime import datetime, timedelta

def demonstrate_live_bedrock():
    print("--- AWS Bedrock: Live Model Invocation ---")
    
    # 1. Initialize Bedrock Runtime Client
    # Region must support the model (e.g., us-east-1 or us-west-2)
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Write a one-sentence summary of why MLOps matters."}]}
        ]
    }

    try:
        print(f"\n[Action] Invoking Claude 3 Haiku ({model_id})...")
        response = client.invoke_model(
            body=json.dumps(payload),
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        print(f"\n✅ Bedrock Output: {response_body['content'][0]['text']}")
        print(f"Usage: {response_body.get('usage')}")

    except Exception as e:
        print(f"\n❌ Bedrock Call Failed: {e}")
        print("Note: Ensure your AWS account has enabled access to Claude 3 models in the Bedrock console.")

def demonstrate_live_costs():
    print("\n--- AWS Cost Explorer: Live Usage Query ---")
    
    client = boto3.client('ce', region_name='us-east-1')
    
    # Define a 3-day window
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    total = 0
    print(f"\n[Action] Querying SageMaker costs from {start_date} to {end_date}...")
    
    try:
        response = client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                "Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}
            }
        )
        
        results = response.get('ResultsByTime', [])
        for day in results:
            amount = day['Total']['UnblendedCost']['Amount']
            unit = day['Total']['UnblendedCost']['Unit']
            print(f"  Day {day['TimePeriod']['Start']}: {amount} {unit}")
            total += float(amount)
        print(f"\n✅ Total SageMaker Cost: {total:.2f} USD")
            
    except Exception as e:
        print(f"❌ Cost Explorer Call Failed: {e}")
        print("Note: The IAM role running this script needs 'ce:GetCostAndUsage' permissions.")

if __name__ == "__main__":
    demonstrate_live_bedrock()
    demonstrate_live_costs()
