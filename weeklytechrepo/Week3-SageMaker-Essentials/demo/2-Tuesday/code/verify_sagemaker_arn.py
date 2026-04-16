"""
Utility: SageMaker Role Validator
This script verifies if a given IAM Role ARN is properly configured for SageMaker.
It checks for:
1. Role existence in your AWS account.
2. Trust relationship with 'sagemaker.amazonaws.com'.
3. Essential permissions (optional check).

Usage:
    python verify_sagemaker_arn.py --arn <YOUR_ROLE_ARN>
"""

import boto3
import argparse
import sys
import json
from botocore.exceptions import ClientError

def print_status(message, success=True):
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")

def verify_role(role_arn):
    print(f"\n--- ⚡ SageMaker Role Validation: {role_arn} ---\n")
    
    # 1. Parse Role Name from ARN
    try:
        if not role_arn.startswith("arn:aws:iam::"):
            print_status("Invalid ARN format (must start with arn:aws:iam::)", False)
            return
        
        role_name = role_arn.split("/")[-1]
    except Exception as e:
        print_status(f"Error parsing ARN: {str(e)}", False)
        return

    iam = boto3.client('iam')
    
    # 2. Check if Role exists
    try:
        response = iam.get_role(RoleName=role_name)
        role = response['Role']
        print_status(f"Role '{role_name}' found in account.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print_status(f"Role '{role_name}' does not exist.", False)
        else:
            print_status(f"Error accessing IAM: {str(e)}", False)
        return

    # 3. Check Trust Relationship
    trust_policy = role.get('AssumeRolePolicyDocument', {})
    statements = trust_policy.get('Statement', [])
    
    is_trusted = False
    for statement in statements:
        principal = statement.get('Principal', {})
        service = principal.get('Service', '')
        
        # Service can be a string or a list
        if isinstance(service, list):
            if 'sagemaker.amazonaws.com' in service:
                is_trusted = True
        elif service == 'sagemaker.amazonaws.com':
            is_trusted = True
            
    if is_trusted:
        print_status("Trust Relationship: SageMaker is a trusted entity.")
    else:
        print_status("Trust Relationship: SageMaker is NOT trusted! (Missing sagemaker.amazonaws.com)", False)
        print("\n   ⚠️  Fix: Edit the Role in AWS Console -> Trust Relationships -> Add 'sagemaker.amazonaws.com'\n")

    # 4. Check for Managed Policies (Basic Check)
    try:
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)
        policies = [p['PolicyName'] for p in attached_policies.get('AttachedPolicies', [])]
        
        if 'AmazonSageMakerFullAccess' in policies:
            print_status("Permission: Found AmazonSageMakerFullAccess.")
        else:
            print(f"   ℹ️  Attached Policies: {', '.join(policies) if policies else 'None'}")
            print("   ⚠️  Warning: Consider attaching 'AmazonSageMakerFullAccess' for full functionality.")
    except Exception:
        print("   ℹ️  Note: Could not list attached policies (check your local IAM permissions).")

    print("\n--- Validation Complete ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a SageMaker Execution Role.")
    parser.add_argument("--arn", required=True, help="The Full ARN of the IAM Role")
    
    # If no args, maybe try to get it from environment or prompt
    if len(sys.argv) == 1:
        print("Usage: python verify_sagemaker_arn.py --arn <YOUR_ROLE_ARN>")
        sys.exit(1)
        
    args = parser.parse_args()
    verify_role(args.arn)
