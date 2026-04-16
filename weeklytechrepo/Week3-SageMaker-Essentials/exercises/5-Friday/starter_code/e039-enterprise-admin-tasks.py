import json
import time
import random
from unittest.mock import MagicMock

# =====================================================================
# MOCK AWS INFRASTRUCTURE (Do not edit this section)
# =====================================================================
class MockBedrockRuntime:
    """
    Simulates boto3.client('bedrock-runtime').
    Raises ThrottlingException after 3 successful calls to test your backoff logic.
    """
    def __init__(self, call_count_limit=3):
        self._call_count = 0
        self._call_count_limit = call_count_limit

    def invoke_model(self, body, modelId, accept, contentType,
                     guardrailIdentifier=None, guardrailVersion=None):
        self._call_count += 1

        # Simulate throttling after N calls
        if self._call_count > self._call_count_limit:
            from botocore.exceptions import ClientError
            error = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
            raise ClientError(error, 'InvokeModel')

        payload = json.loads(body)
        messages = payload.get('messages', [{}])
        prompt_text = messages[0].get('content', 'N/A') if messages else 'N/A'

        fake_body = json.dumps({
            "content": [{"text": f"[Bedrock Response #{self._call_count}] Summary: '{prompt_text[:40]}...'"}],
            "usage": {"input_tokens": len(prompt_text.split()), "output_tokens": 42}
        })
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, key: MagicMock(
            read=lambda: fake_body.encode()
        ) if key == 'body' else None
        return mock_response


class MockCostExplorer:
    """Simulates boto3.client('ce')."""
    def get_cost_and_usage(self, TimePeriod, Granularity, Filter, Metrics):
        tag_value = Filter.get('Tags', {}).get('Values', ['Unknown'])[0]
        print(f"[Cost Explorer] Querying costs for tag value: '{tag_value}'")
        return {
            "ResultsByTime": [{
                "TimePeriod": TimePeriod,
                "Total": {
                    "UnblendedCost": {"Amount": "142.87", "Unit": "USD"}
                }
            }]
        }


# =====================================================================
# YOUR TASKS
# =====================================================================
def invoke_bedrock_model(bedrock_client, prompt: str) -> dict:
    """
    Task 1: Invoke a Bedrock Foundation Model via Boto3.
    """
    print(f"\n--- Invoking Bedrock ---")
    print(f"Prompt (first 60 chars): '{prompt[:60]}...'")

    # TODO 1: Build the request payload dictionary with:
    # - "anthropic_version": "bedrock-2023-05-31"
    # - "max_tokens": 200
    # - "temperature": 0.5
    # - "messages": [{"role": "user", "content": prompt}]
    payload = None

    # TODO 2: Call bedrock_client.invoke_model() with:
    # - body=json.dumps(payload)
    # - modelId="anthropic.claude-3-haiku-20240307-v1:0"
    # - accept="application/json"
    # - contentType="application/json"
    response = None

    if response:
        # TODO 3: Parse the response body:
        # result = json.loads(response['body'].read())
        result = None
        if result:
            text = result['content'][0]['text']
            usage = result.get('usage', {})
            print(f"Response: {text}")
            print(f"Token usage: {usage}")
            return result
    return None


def invoke_with_backoff(bedrock_client, prompt: str, max_retries: int = 5) -> dict:
    """
    Task 2: Wrap invoke_bedrock_model with exponential backoff for ThrottlingException.
    """
    # TODO: Loop up to max_retries times.
    # On each attempt, call invoke_bedrock_model(bedrock_client, prompt).
    # If a ClientError with Code 'ThrottlingException' is raised:
    #   - If attempts remain, calculate: wait_time = (2 ** attempt) + random.uniform(0, 1)
    #   - Print a warning and call time.sleep(wait_time)
    #   - Continue to the next attempt
    # For any other ClientError, re-raise immediately (don't retry unknown errors).
    # After max_retries exhausted, raise RuntimeError("Max retries exceeded.")
    pass  # Replace this with your implementation


def get_project_spend(ce_client, tag_key: str, tag_value: str) -> str:
    """
    Task 3: Query AWS Cost Explorer for project spend filtered by a resource tag.
    """
    print(f"\n--- Querying Cost Explorer ---")
    import datetime
    today = datetime.date.today()
    first_of_month = today.replace(day=1).isoformat()

    # TODO 1: Call ce_client.get_cost_and_usage() with:
    # - TimePeriod: {"Start": first_of_month, "End": today.isoformat()}
    # - Granularity: "MONTHLY"
    # - Filter: {"Tags": {"Key": tag_key, "Values": [tag_value]}}
    # - Metrics: ["UnblendedCost"]
    response = None

    if response:
        # TODO 2: Extract and print the total cost:
        # amount = response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
        amount = None
        if amount:
            print(f"Total spend for '{tag_value}': ${amount} USD")
            return amount
    return "0.00"


def simulate_cloudwatch_alarm(latency_ms: float, threshold_ms: float = 500.0):
    """
    Task 4: Evaluate a simulated CloudWatch latency metric against a threshold.
    """
    print(f"\n--- CloudWatch Monitor | ModelLatency = {latency_ms}ms ---")

    # TODO: Compare latency_ms to threshold_ms.
    # If latency_ms > threshold_ms:
    #   print("🚨 ALARM: ModelLatency exceeded threshold! Investigate endpoint scaling.")
    # Else:
    #   print("✅ OK: ModelLatency within acceptable range.")
    pass  # Replace this with your implementation


# =====================================================================
# SIMULATION RUNNER
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise Admin Tasks — Bedrock & Costs Lab")
    print("=" * 60)

    # Mock clients (replace with real boto3 clients in AWS environment)
    bedrock = MockBedrockRuntime(call_count_limit=3)
    ce = MockCostExplorer()

    PROMPT = "Summarize the key benefits of MLOps pipelines in two sentences."

    # Tasks 1 & 2: Invoke with throttling protection
    # The mock client will throttle after 3 calls — your backoff should handle this.
    print("\n[Testing Tasks 1 & 2: Bedrock invocation with backoff]")
    result = invoke_with_backoff(bedrock, PROMPT, max_retries=5)
    if result:
        print("Backoff invocation succeeded.")

    # Task 3: Cost Explorer query
    print("\n[Testing Task 3: Cost Explorer]")
    spend = get_project_spend(ce, tag_key="Project", tag_value="Project-Secure-AI")
    if spend:
        print(f"Monthly spend confirmed: ${spend}")

    # Task 4: CloudWatch alarm simulation
    print("\n[Testing Task 4: CloudWatch Alarm]")
    simulate_cloudwatch_alarm(latency_ms=620.0)   # Should trigger alarm
    simulate_cloudwatch_alarm(latency_ms=210.0)   # Should be OK
