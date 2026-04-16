# Demo: SageMaker Ecosystem and Script Mode

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Managed Cloud Service** | *"What's the difference between hiring a contractor to build a house vs. renting a house that's already built and maintained by a landlord? How does that map to local servers vs. managed cloud services?"* |
| **IAM Role** | *"If a new employee joins a company, they shouldn't automatically have access to the CEO's email, HR salary data, and engineering code. What process determines exactly what they can access? What's the cloud equivalent?"* |
| **S3 Bucket** | *"Where would you store 500GB of training images so that an EC2 server in AWS could access them instantly without downloading from the internet? What kind of storage lives close to AWS compute?"* |
| **Docker Container** | *"Why would a Python script that works on your laptop sometimes fail on a server? What technology could lock in the exact Python version, library versions, and OS to guarantee consistency?"* |

## Phase 1: The Concept (Whiteboard/Diagram)

**Time:** 15 mins

1. Open `diagrams/sagemaker-ecosystem.mermaid`.
2. Flow overview: Trace the path from the local laptop to the `Estimator`. Explain that the Estimator is just configuration code; nothing runs locally.
3. Walk through the **AWS Cloud Ecosystem** subgraph. Show how the Estimator provisions a VM, pulls an Amazon-managed Docker image (so you don't have to write Dockerfiles), downloads your PyTorch code, and executes it.
4. **Discussion:** Point to the **AWS IAM Role** node. Ask the class: "Why does the EC2 instance need explicit 'Read' permissions for S3?" (Answer: Without the role, the instance has zero rights to access your proprietary data bucket).

## Phase 2: The Code (Live Implementation)

**Time:** 20 mins

1. Open `code/d049-sagemaker-studio-and-jumpstart.py`.
2. Begin with **Part 2: The SageMaker Host SDK**.
   - Walk through the `PyTorch()` estimator instantiation.
   - *Note: Emphasize the separation of concerns. The `instance_type` controls the hardware costs, while the `framework_version` guarantees the library versions.*
3. Transition to **Part 1: The Local Script Mode Entry Point**.
   - Explain how `argparse` acts as the bridge. The `hyperparameters` dictionary in the Estimator is automatically converted into bash flags (`--epochs 20`) when SageMaker runs the script.
   - Point out the `os.environ.get('SM_MODEL_DIR')` pattern. This is how SageMaker tells the script where to save the final artifact so AWS can seamlessly copy it to S3 before terminating the instance.
4. Execute the script to show the configuration and simulation outputs.

## Phase 3: SageMaker JumpStart (No-Code/Low-Code)

**Time:** 15 mins

1. **The UI Path:** Open `walkthrough.md`.
   - Walk through the visual steps of selecting a model in the JumpStart Hub.
   - Discuss why JumpStart is preferred for rapid prototyping (transfer learning with 1-click).
2. **The Code Path:** Open `code/d049b-jumpstart-finetune-deploy.py`.
   - Show how the `JumpStartEstimator` automates the configuration that was manual in Script Mode (`d049`).
   - Mention that JumpStart handles the Docker image selection and training scripts internally.
3. **Discussion:** "When would you use Script Mode (`d049`) vs. JumpStart (`d049b`)?" (Answer: JumpStart for common models/tasks to save time; Script Mode for highly custom architectures or research).

## Summary

Reiterate that SageMaker provides multiple entry points—from pure code (Script Mode) to pre-built solutions (JumpStart)—allowing teams to choose the right level of abstraction for their project.
