# e052: Final Capstone Presentation Rubric

## Objective
The final exercise is to present your fully functional LangGraph multi-agent pipeline to the class. Ensure that the logic pathways adhere strictly to the curriculum principles for agentic generation, resilience, and scalable deployment.

## Presentation Rubric

**1. Enterprise Context (10%)**
- Clearly articulate the specific business problem the AI pipeline solves.
- Describe why standard RAG was insufficient and why an Agent based looping structure was required computationally.

**2. Architecture Validation (40%)**
- Project a visualization of your system flow.
- Highlight the exact logic triggers used by your Supervisor or Orhcestrator modules to delegate context routing.
- Prove that your nodes reliably append conversational State safely via Reducers or Checkpointers.

**3. Infrastructure-As-Code Deployment (30%)**
- Walk through your Serverless Application Model (SAM) `template.yaml`.
- Prove your local agent has been securely wrapped via AWS Lambda to accept inbound requests over API Gateway.
- Highlight how `bedrock:InvokeModel` IAM permissions are correctly sandboxed.

**4. Safety & Human Integrations (20%)**
- Document an edge-case trigger you established preventing agent hallucinations using an EDD Golden Dataset test trace.
- Demonstrate an explicit `interrupt()` checkpoint inserted before a volatile action (like external database alteration) proving a Human-in-The-Loop capability.
