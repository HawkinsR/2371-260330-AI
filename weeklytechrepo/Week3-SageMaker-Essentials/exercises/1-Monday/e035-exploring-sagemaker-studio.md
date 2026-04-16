# Lab: Exploring SageMaker Studio

## The Scenario

Your organization is moving to AWS SageMaker for collaborative machine learning. As an AI Engineer, your first task is to orient within the **SageMaker Studio** environment, understand how **User Profiles** and **IAM Execution Roles** permit access to cloud resources, and test a pre-built foundation model using **SageMaker JumpStart**.

## Core Tasks


1. **IAM and Studio Setup:**
   - Define the purpose of a **Studio Domain** and a **User Profile**.
   - Explain how an **IAM Execution Role** (e.g., `AmazonSageMakerFullAccess`) allows SageMaker to interact with S3 and CloudWatch.
2. **JumpStart Exploration:**
   - Locate the **SageMaker JumpStart** icon in the Studio UI.
   - Search for a popular foundation model (e.g., **Llama 3** or **Mistral 7B**).
3. **Prompt Engineering Test:**
   - Deploy a temporary JumpStart model endpoint (simulated).
   - Test a **Zero-Shot** vs. **Few-Shot** prompt in a Studio notebook.
   - Analyze the difference in output quality between a simple query and one with provided examples.

## Definition of Done

- You can explain the hierarchy of Domain -> Profile -> IAM Role.
- You have identified a JumpStart model suitable for text summarization.
- You have recorded the results of a Few-Shot prompting test comparing model accuracy.
