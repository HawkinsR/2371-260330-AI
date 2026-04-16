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

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    if response:
        result = json.loads(response['body'].read())
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
    for attempt in range(max_retries):
        try:
            return invoke_bedrock_model(bedrock_client, prompt)
        except Exception as e:
            # Handle ClientError thrown by boto3 simulations
            if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == 'ThrottlingException':
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Throttled. Backing off for {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e # Real error we shouldn't catch
    raise RuntimeError("Max retries exceeded.")


def get_project_spend(ce_client, tag_key: str, tag_value: str) -> str:
    """
    Task 3: Query AWS Cost Explorer for project spend filtered by a resource tag.
    """
    print(f"\n--- Querying Cost Explorer ---")
    import datetime
    today = datetime.date.today()
    first_of_month = today.replace(day=1).isoformat()

    response = ce_client.get_cost_and_usage(
        TimePeriod={"Start": first_of_month, "End": today.isoformat()},
        Granularity="MONTHLY",
        Filter={"Tags": {"Key": tag_key, "Values": [tag_value]}},
        Metrics=["UnblendedCost"]
    )

    if response:
        amount = response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
        if amount:
            print(f"Total spend for '{tag_value}': ${amount} USD")
            return amount
    return "0.00"


def simulate_cloudwatch_alarm(latency_ms: float, threshold_ms: float = 500.0):
    """
    Task 4: Evaluate a simulated CloudWatch latency metric against a threshold.
    """
    print(f"\n--- CloudWatch Monitor | ModelLatency = {latency_ms}ms ---")

    if latency_ms > threshold_ms:
        print("🚨 ALARM: ModelLatency exceeded threshold! Investigate endpoint scaling.")
    else:
        print("✅ OK: ModelLatency within acceptable range.")


# =====================================================================
# SIMULATION RUNNER
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise Admin Tasks — Bedrock & Costs Lab")
    print("=" * 60)

    bedrock = MockBedrockRuntime(call_count_limit=3)
    ce = MockCostExplorer()

    PROMPT = "Summarize the key benefits of MLOps pipelines in two sentences."

    print("\n[Testing Tasks 1 & 2: Bedrock invocation with backoff]")
    result = invoke_with_backoff(bedrock, PROMPT, max_retries=5)
    if result:
        print("Backoff invocation succeeded.")
        
    print("\n[Testing Task 2 Throttling: Bedrock invocation with backoff overload]")
    try:
        # Pushing the client past the limit of 3 to test backoff logic
        invoke_with_backoff(bedrock, PROMPT, max_retries=5)
        invoke_with_backoff(bedrock, PROMPT, max_retries=5)
        invoke_with_backoff(bedrock, PROMPT, max_retries=5)
    except Exception as e:
        print(f"Exception generated correctly: {e}")

    print("\n[Testing Task 3: Cost Explorer]")
    spend = get_project_spend(ce, tag_key="Project", tag_value="Project-Secure-AI")
    if spend:
        print(f"Monthly spend confirmed: ${spend}")

    print("\n[Testing Task 4: CloudWatch Alarm]")
    simulate_cloudwatch_alarm(latency_ms=620.0)   # Should trigger alarm
    simulate_cloudwatch_alarm(latency_ms=210.0)   # Should be OK
