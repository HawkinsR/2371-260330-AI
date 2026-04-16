# Demo: Boto3, Bedrock, and Cost Management

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **AWS Bedrock** | *"If you wanted to use Claude or Llama 3 in your enterprise app without provisioning a single GPU server or training a model, what would you need? What's the tradeoff between calling a Bedrock API vs. hosting your own model?"* |
| **API Throttling** | *"If 1,000 engineers in your company all hit the Bedrock API simultaneously, AWS can't serve every request at once. What HTTP error would you receive? What's the right response — retry immediately, or wait? Why?"* |
| **Exponential Backoff** | *"If retrying immediately after a throttle just contributes to the overload, what's a smarter retry strategy? If your first wait is 2 seconds, how long should the second wait be? The third? What's the formula?"* |
| **AWS Cost Explorer** | *"Your team just got a surprise $8,000 bill from AWS. How would you find out which service caused it and which team is responsible? What AWS feature breaks costs down by resource tag?"* |
| **CloudWatch** | *"Your SageMaker endpoint is responding slowly. How would you know if it's a model latency problem or a cold-start infrastructure problem — without SSH-ing into the server?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/bedrock-and-costs.mermaid`.
2. Trace the request path: Local Boto3 SDK → Bedrock Runtime API → Foundation Model (Claude / Llama) → CloudWatch metric emission → Cost Explorer aggregation.
3. Walk through the **Cost Management** subgraph: tagging resources at creation → Cost Explorer aggregating by tag → budget alert thresholds.
4. **Discussion:** Ask the class: *"AWS charges per 1,000 tokens for Bedrock. If your Tuesday input validation is bypassed and a prompt injection asks the model to write a 10,000-word essay — what happens to the bill?"* (Answer: Costs spike unexpectedly. This is why Tuesday's `validate_prompt_input()` is a cost control mechanism, not just a security measure.)

## Phase 2: The Code (Live Implementation)
**Time:** 25 mins
1. Open `code/d053-boto3-bedrock-and-costs.py`.
2. Walk through `demonstrate_bedrock_and_costs()`:
   - Highlight the `invoke_model()` parameters: `modelId`, `body` (JSON), `accept`, `contentType`. Emphasize that the `body` payload format is **Anthropic-specific** — different Bedrock providers (Mistral, Llama, Titan) use entirely different schemas.
   - Show the `temperature` parameter. Ask: *"What would `temperature=0.0` give you vs. `temperature=1.0`?"* (Answer: Deterministic/reproducible vs. creative/random — critical for production vs. experimental use.)
   - Walk the response parsing chain: `response.get('body').read()` → `json.loads()` → `content[0]['text']`.
   - Show the Cost Explorer `Filter` dictionary. Emphasize that **tagging is not optional in enterprise** — without tags, you cannot attribute costs to specific projects or teams for billing chargebacks.
3. Walk through `demonstrate_throttling_and_backoff()`:
   - Show the `MockBedrockRuntime(call_count_limit=3)` constructor. Point out that the 4th call raises a `MockClientError` with code `ThrottlingException`.
   - Walk the retry loop. Ask: *"Why does `wait_time = (2 ** attempt) + random.uniform(0, 1)`? What is the `random.uniform` doing?"* (Answer: **Jitter** — if 1,000 clients all backed off by exactly 2s, 4s, 8s, they'd all retry simultaneously. Random jitter desynchronizes the thundering herd.)
   - Point out the `else` branch that re-raises non-throttling `ClientError`s immediately. Ask: *"Why is it important NOT to retry a permissions error (403) or a wrong-region error?"* (Answer: Retrying won't fix a credential or config problem — it just wastes time and money.)
   - Execute the script. Watch the `⚠️ ThrottlingException — waiting Xs` lines appear on calls 4 and 5.

## Summary
Reiterate that enterprise Bedrock integration requires three defensive layers working together: **input validation** (Tuesday), **throttling resilience** (today's backoff demo), and **cost accountability** through resource tagging and CloudWatch monitoring. A production AI system missing any one of these is a financial and security liability.
