# Demo: ReAct Prompting and Custom Estimators

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **ReAct (Reasoning + Acting)** | *"If you asked a junior developer to 'deploy the model' and they just did it without checking the metrics first, would you trust that? What's the difference between blindly acting and reasoning about why you're acting before you do it?"* |
| **MLOps** | *"What does DevOps do for software development (automated testing, deployment pipelines, monitoring)? Now imagine doing the same for a model that trains for 12 hours on 100GB of data. What extra challenges arise?"* |
| **Environment Variable** | *"If you hardcoded the path `/Users/richard/data/train/` in your script and SageMaker runs it on a Linux server, what would happen? How would an environment variable solve this?"* |
| **Hyperparameters** | *"What's the difference between the parameters a model learns during training (weights) and the parameters we choose before training begins (learning rate, batch size)? Why would you want to pass the second type from outside the script?"* |
| **Asynchronous Execution** | *"When you send an email, does your client freeze until the recipient reads and replies? How does asynchronous execution apply to launching a multi-hour training job in the cloud?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 15 mins
1. Open `diagrams/estimator-training-job.mermaid`.
2. **Bridge from Yesterday's Prompting:** Draw the ReAct loop on the whiteboard:
   - `Thought → Action → Observation → Thought → ...`
   - Today's **Action** is: *Launch a SageMaker Estimator.* The **Observation** is: *monitoring CloudWatch logs to confirm the loss is decreasing.*
3. Trace the path from the local `launch_job.py` through the `.fit()` call into the cloud.
4. Walk through the **Asynchronous Cloud Execution** subgraph. Emphasize that all four steps (uploading, provisioning, downloading, executing) happen entirely hands-off after `.fit()` is called.
5. **Discussion:** Ask the class: *"If your training data is 500GB, why is it better to rely on `estimator.fit()` doing an S3 download into the container rather than parsing it locally on your laptop?"* (Answer: Networking within AWS between S3 and EC2 is massively faster than your local ISP, and the cloud instance can hold the data in memory without crashing your laptop.)

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d050-advanced-prompting-and-estimations.py`.
2. Walk through `demonstrate_react_prompting()` **first**:
   - Show the three-item `react_sequence` list. Each dictionary is one `Thought → Action → Observation` triple.
   - Emphasize that **each triple maps to a real workflow decision**: analyzing requirements → launching the training job → monitoring the results.
   - *Note: Ask the class — "If the Observation in Step 2 showed training loss was increasing instead of decreasing, what would the next Thought be?"* (Answer: Diagnose the hyperparameters — lower LR, different optimizer, check the data pipeline.)
   - Highlight the "auditable reasoning" framing. This satisfies enterprise AI governance requirements where every AI-driven action must have a documented justification.
3. Transition to `demonstrate_estimator_launch()`. This simulates the real **Action** step from the ReAct sequence:
   - Show the `hyperparameters` dictionary and how it maps into `argparse` flags inside the container script.
   - Highlight `metric_definitions` and the `Loss: ([0-9\\.]+)` regex. Ask: *"How does SageMaker know what number to pull from the line `Epoch 1/3 - Loss: 1.0000`?"* (Answer: The regex captures group 1 — the digit string after `Loss: `.)
4. Execute the script. Walk the class through the terminal transition from local laptop logic into the `>>> AWS SAGEMAKER CONTAINER INITIALIZING <<<` block.

## Phase 3: The Full Pipeline (d050b)
**Time:** 25 mins
1. Open `code/d050b-custom-train-deploy.py`.
2. Explain that while `d050` was just a launch demo, `d050b` is the **Full Journey**:
   - **Training**: Show `src/train_linear.py` and the `USE_GPU` toggle. Explain how SageMaker handles the infrastructure swap.
   - **Checkpointing**: Highlight the "Interlude" where we see the artifact in S3. 
   - **Deployment**: Walk through `src/inference.py` and how `predictor.deploy()` stands up a live REST API.
3. Start the script early, as the endpoint deployment takes time. Use the deployment wait-time to revisit the cost-alert cleanup code at the bottom of the script.

## Phase 4: Beyond the Training Job (d050c)
**Time:** 20 mins
1. Open `code/d050c-local-model-upload.py`.
2. **The "Hybrid" Workflow**: Explain that this script is specifically designed to bridge the gap between a trainee's home laptop and the AWS cloud.
3. **Local PC Prerequisites (Discussion)**:
   - Ask: *"If you're running code on your own laptop, how does AWS know who you are?"* (Answer: AWS CLI configuration / `aws configure`.)
   - Highlight the **Execution Role ARN**. Explain that since a home laptop doesn't have an IAM Role attached, we must explicitly pass the ARN of the role we want SageMaker to use in the cloud.
4. **Manual Packaging Walkthrough**:
   - Show how the script uses Python's `tarfile` to manually zip `model.pth` and `inference.py`. 
   - Point out the `sagemaker.Session().upload_data()` command. This is the **Bridge to the Cloud** for existing files.
5. **Deployment & Execution**:
   - Run the script. If in Studio, show the automatic role detection. If time permits, discuss how they would paste their own Role ARN if they were at home.
   - Emphasize that once the model is in S3, the deployment process is identical to Phase 3.

## Summary
Reiterate that ReAct transforms a raw training job from a black box into an auditable, step-by-step decision chain, and that the PyTorch Estimator is the cloud infrastructure primitive that makes the "Action" step possible at enterprise scale — connecting AI reasoning directly to real compute. The transition from CPU to GPU via a single toggle demonstrates the true power of cloud-native MLOps.
