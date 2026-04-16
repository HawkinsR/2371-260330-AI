# AWS Bedrock and Cost Management

## Learning Objectives

- Orient within the AWS Bedrock Ecosystem and available Foundation Models.
- Utilize the Bedrock Runtime API and understand key Inference Parameters.
- Configure CloudWatch Monitoring to track model invocation and latency.
- Implement Cost Management strategies and Tagging policies for SageMaker resources.
- Handle API Quotas and Throttling using best practices for enterprise scale.

## Why This Matters

As you move from prototyping to enterprise deployment, two factors become dominant: **Safety** and **Cost**. AWS Bedrock provides a serverless way to access the world's most powerful models, but without monitoring, costs can spiral out of control. Mastering the Boto3 SDK for Bedrock and integrating CloudWatch monitoring ensures that your AI applications are not only intelligent but also economically sustainable and secure.

## The Concept

### AWS Bedrock Orientation

Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like Anthropic, Meta, Mistral, and Amazon via a single API. Unlike SageMaker, Bedrock is **serverless**, meaning you don't manage instances or clusters. You pay only for the tokens you consume.

### CLI and Boto3 SDK Setup

Before any Bedrock or SageMaker SDK code can run, your local environment must be configured with AWS credentials. The standard approach is the **AWS Shared Credentials File**, populated using the AWS CLI:

```bash
# Install the AWS CLI (if not already installed)
pip install awscli

# Configure credentials interactively
# You will be prompted for your Access Key ID, Secret Key, region, and output format
aws configure
# AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name [None]: us-east-1
# Default output format [None]: json
```

This writes credentials to `~/.aws/credentials` and configuration to `~/.aws/config`. The `boto3` library automatically reads from these files when you instantiate a client without explicit credentials.

> **Key Term - AWS Shared Credentials File:** A local file (`~/.aws/credentials`) that stores AWS Access Key ID and Secret Access Key pairs. The `boto3` SDK reads this file automatically to authenticate API requests, eliminating the need to hardcode secrets in source code (which is a major security vulnerability).

### Inference Parameters (Bedrock Runtime)

When calling a model via Bedrock, you can fine-tune its behavior using inference parameters:

> **Key Term - Temperature:** Controls output randomness. A value of `0.0` makes the model deterministic — always choosing the highest-probability next token. A value of `1.0` introduces maximum randomness. For factual or structured tasks, use `temperature ≤ 0.2`.

> **Key Term - Top-P (Nucleus Sampling):** The model restricts its token selection to the smallest set of candidates whose combined probability reaches `top_p`. Lower values (e.g., `0.5`) produce tighter, more predictable outputs.

> **Key Term - Max Tokens:** The maximum number of tokens the model will generate in its response. Capping this prevents runaway responses and controls API cost directly (since most providers bill per output token).

> **Key Term - Stop Sequences:** Custom strings that signal the model to stop generating immediately. For example, `"\nUser:"` can be used to prevent a chat model from role-playing both sides of a conversation.

> **Key Term - Bedrock Runtime API:** The specific AWS API used to invoke models and receive predictions. It is separate from the control plane API used to manage model access and settings.

### Cost Management and CloudWatch

AI can be expensive. To keep costs under control:
1.  **Tagging:** Always apply cost-allocation tags to your SageMaker endpoints and Bedrock calls.
2.  **CloudWatch Metrics:** Monitor `Invocations`, `ModelLatency`, and `OverheadLatency`. 
3.  **Throttling:** AWS imposes quotas on how many tokens per minute (TPM) you can process. Your code must handle `ThrottlingException` errors gracefully using exponential backoff.

> **Key Term - CloudWatch:** The central monitoring and logging service for AWS. It collects metrics from SageMaker and Bedrock, allowing you to set alarms if costs exceed a budget or if latency becomes too high.

> **Key Term - Throttling:** A mechanism to limit the rate of API requests. If you exceed your assigned quota, AWS will reject further requests until the timer resets. Handling this requires "Backoff and Retry" logic in your application.

### Handling Throttling: Exponential Backoff

When your application hits an API quota limit, AWS returns a `ThrottlingException`. The correct response is **exponential backoff**: wait, then retry. Each retry doubles the wait time, with a small random **jitter** added to prevent multiple clients from retrying simultaneously and creating another spike.

```python
import boto3
import time
import random
from botocore.exceptions import ClientError

def invoke_with_backoff(bedrock_client, model_id: str, payload: dict,
                        max_retries: int = 5, base_delay: float = 1.0):
    """Invoke a Bedrock model with exponential backoff on throttling."""
    import json
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                body=json.dumps(payload),
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )
            return json.loads(response['body'].read())

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, 8s... + random jitter
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Throttled. Retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise  # Re-raise if max retries exceeded or different error

    raise RuntimeError("Max retries exceeded.")
```

### Bedrock Guardrails

**Bedrock Guardrails** is a managed content filtering layer you configure in the AWS console and attach to your Bedrock API calls. Unlike prompt-level instructions (which a clever user can override), Guardrails are enforced server-side by AWS before the response reaches your application.

Guardrails can:
- **Block harmful topics** — define categories (violence, illegal activity) that the model is prohibited from discussing.
- **Redact PII** — automatically detect and mask names, SSNs, phone numbers in both inputs *and* outputs.
- **Filter profanity and hate speech** with configurable severity thresholds.
- **Apply word filters** — block specific competitor names, proprietary terms, or legal no-go phrases.

You apply a Guardrail to an API call by including its ID in the request:

```python
# Attach a pre-configured Guardrail to a Bedrock invocation
response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    accept="application/json",
    contentType="application/json",
    guardrailIdentifier="my-guardrail-id",   # From AWS console
    guardrailVersion="DRAFT"                  # or a specific published version
)
```

> **Key Term - Bedrock Guardrails:** A server-side content policy layer in AWS Bedrock that filters both incoming prompts and outgoing model responses before they reach user-facing applications. Enforced by AWS infrastructure, not by the prompt, making them resistant to prompt injection attempts.

## Code Example

```python
import boto3
import json

# 1. Initialize the Bedrock Runtime client
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# 2. Define the payload with inference parameters
prompt_data = "Write a haiku about a cloud-based GPU."

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 100,
    "temperature": 0.5,
    "messages": [
        {"role": "user", "content": prompt_data}
    ]
}

# 3. Invoke the model via Boto3
response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    accept="application/json",
    contentType="application/json"
)

# 4. Parse and display the result
response_body = json.loads(response.get('body').read())
print(response_body['content'][0]['text'])
```

## Additional Resources

- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [Boto3 Bedrock Runtime Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
- [Tagging AWS Resources](https://docs.aws.amazon.com/general/latest/gr/aws_tagging.html)
