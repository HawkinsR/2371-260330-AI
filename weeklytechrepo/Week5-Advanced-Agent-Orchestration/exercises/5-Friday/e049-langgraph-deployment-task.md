# E049: LangGraph Production Deployment Task

## Objective
Convert your local LangGraph AI agent into a serverless AWS application utilizing AWS Serverless Application Model (SAM) templates. You will construct the IaC (Infrastructure as Code) skeleton to launch your API Gateway and lambda environment.

## Instructions
1. Open the `starter_code/` folder containing your `template.yaml` skeleton and `app.py`.
2. Edit `template.yaml`.
3. Add the proper `Runtime` property for a Python 3.11 lambda deployment.
4. Inject the `Policies` required to invoke `bedrock:InvokeModel`. Without this IAM role, your Lambda graph execution will fail.
5. In `app.py`, update the mock state machine to invoke the authentic `langchain_aws.ChatBedrock` instance. 
6. Using your local terminal, run `sam validate` (if installed) or mentally trace the architecture to verify it matches the solution.
