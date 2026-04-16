#!/bin/bash
set -e

echo "=== AWS SAM Deployment: LangGraph Agent ==="
echo "1. Building the SAM application..."
sam build

echo "2. Deploying to AWS..."
echo "(This will provision the Lambda function, API Gateway, and IAM Bedrock Roles.)"
sam deploy --guided

echo "Deployment complete! You can test your endpoint using:"
echo 'curl -X POST <API_URL> -H "Content-Type: application/json" -d "{\"message\": \"What is LangGraph?\"}"'
