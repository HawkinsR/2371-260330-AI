# Demo: SageMaker Estimators and Training

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **MLOps** | *"What does DevOps do for software development (automated testing, deployment pipelines, monitoring)? Now imagine doing the same for a model that trains for 12 hours on 100GB of data. What extra challenges arise?"* |
| **Environment Variable** | *"If you hardcoded the path `/Users/richard/data/train/` in your script and SageMaker runs it on a Linux server, what would happen? How would an environment variable solve this?"* |
| **Hyperparameters** | *"What's the difference between the parameters a model learns during training (weights) and the parameters we choose before training begins (like learning rate or batch size)? Why would you want to pass the second type from outside the script?"* |
| **Asynchronous Execution** | *"When you send an email, does your email client freeze until the recipient reads and replies? How does asynchronous execution apply to launching a multi-hour training job in the cloud?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/estimator-training-job.mermaid`.
2. Trace the path from the local `'launch_job.py'` script through the `.fit()` command.
3. Walk through the **Asynchronous Cloud Execution** subgraph. Emphasize that all four of these steps (uploading, provisioning, downloading, executing) happen entirely hands-off. 
4. **Discussion:** Ask the class: "If your training data is 500GB, why is it better to rely on `estimator.fit()` doing an S3 download into the container rather than parsing it locally on your laptop?" (Answer: Networking speeds within AWS between S3 and EC2 are massively faster than your local ISP, and an EC2 instance can hold the data in memory without crashing).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d050-launching-custom-estimators.py`.
2. This file simulates two different spaces: the local laptop, and the remote AWS server. Start in **FILE 2** (the local script).
   - Show how the dictionary inside `hyperparameters={'epochs': 50}` maps directly to what we want the Cloud instance to run.
   - Highlight `metric_definitions`. Explain that this Regex is how SageMaker parses raw terminal print statements into pretty CloudWatch graphs.
3. Transition to **FILE 1** (the remote script).
   - Point out `os.environ.get('SM_MODEL_DIR')`. Reiterate that we must blindly trust the path SageMaker gives us.
   - Show the `mock_sys_args` parser logic.
4. Execute the script. 
5. Walk the class through the output, pointing out the distinct line where the simulation switches from the local laptop logic to the `[INSIDE CONTAINER]` logic.

## Summary
Reiterate that mastering Estimators is the core skill of modern MLOps, allowing engineers to write code locally but execute globally on demand.
