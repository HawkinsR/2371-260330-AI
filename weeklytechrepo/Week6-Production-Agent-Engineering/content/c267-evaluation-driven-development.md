# c267: Evaluation Driven Development (EDD)

## The Shift to EDD
Developing non-deterministic LLM pipelines requires a paradigm shift away from traditional Test-Driven Development (TDD). **Evaluation Driven Development (EDD)** acknowledges that exact string-matching fails against standard LLMs; instead, EDD mandates scoring iterations of a prompt or agent against a robust baseline before pushing code to production.

## Unit Testing Agents
When you construct a LangGraph network, you must verify the structural integrity of your nodes independent of the LLM’s stochastic nature.
- Asserting graph structures and edge routing behaviors ensure that hardcoded fallback limits prevent infinite looping.
- Mocking provides immediate test turnaround times to isolate pure logic errors from external latency.

### Mocking Tool Calls
To prevent incurring token costs and executing live (potentially destructive) external API requests during CI/CD test runs, tool calls must be aggressively mocked. 
By overriding the bindings natively via patching constants directly within your `@tool` execution signatures, you verify that your routing functions successfully trigger *the attempt* to call a tool, without interacting with the outside world.

## Evaluation Infrastructure
It is not enough to mock logic; we must mathematically evaluate the quality of the generative output when the full system runs.

### Creating Golden Datasets
A **Golden Dataset** is a curated list of input scenarios paired strictly with idealized output annotations. It acts as the immutable ground truth for a system.
When you edit a supervisor prompt or tweak a vector DB temperature, you stream the entire Golden Dataset against your pipeline to analyze for regression metrics.

### LangSmith Evaluators
Evaluating hundreds of generative answers manually against a Golden Dataset is impossible. LangSmith Evaluators leverage *LLM-as-a-Judge* mechanics:
- **Heuristics**: Does the output strictly abide by the Pydantic JSON schema?
- **Accuracy**: Does the generative answer effectively solve the original user intent outlined in the Golden Dataset reference? 

By executing a `.run_on_dataset()` execution through LangSmith, teams can actively track percentage accuracy across historical software versions automatically.
