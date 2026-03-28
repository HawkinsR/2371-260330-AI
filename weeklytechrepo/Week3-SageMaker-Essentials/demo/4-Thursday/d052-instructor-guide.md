# Demo: SageMaker Endpoints and Scaling

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Endpoint** | *"If your model lives inside a Docker container on an AWS server, how would a React web app on a user's iPhone communicate with it to get a prediction? What needs to exist between them?"* |
| **Auto-scaling** | *"Imagine your model gets featured on a news article and suddenly 10,000 users hit it in 5 minutes. If you only had 1 server provisioned, what would happen? What mechanism prevents that?"* |
| **Blue/Green Deployment** | *"If a new model version is buggy, how do you instantly roll back to the old version without downtime? What would you need to have running *alongside* the new version to make that possible?"* |
| **Cost Optimization** | *"A model endpoint runs 24/7/365. At $2/hour for a GPU server, how much would you spend in a year doing nothing? What feature saves money when traffic is low overnight?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/endpoint-scaling.mermaid`.
2. Trace the path from the raw `model.tar.gz` and the `inference.py` script fusing into the `PyTorchModel` object.
3. Walk through the **Auto-scaling Load Balancer** subgraph. Emphasize that the Client Application never talks directly to "Instance 1". It hits an Elastic Load Balancer (ELB) url, which transparently routes the traffic.
4. **Discussion:** Point to the CloudWatch Alarm. Ask the class: "If an endpoint is scaling from 1 instance to 3 instances during a traffic spike, how does AWS ensure no incoming HTTP requests are dropped while instances 2 and 3 are booting up?" (Answer: The Load Balancer holds the traffic or exclusively routes to the surviving Instance 1 until health checks pass on the new instances. This is a Blue/Green deployment).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d052-endpoint-deployment-and-update.py`.
2. Explain to the class that because SageMaker endpoints bill per minute of uptime, running real infrastructure for a demo is costly. We are using custom Mock classes to demonstrate the exact SDK syntax safely in local memory.
3. Walk through `demonstrate_endpoint_lifecycle()`.
   - Contrast `PyTorchModel` with the `PyTorch` Estimator from yesterday. The Estimator trains models; `PyTorchModel` deploys already-trained models.
   - Show the `.deploy()` call. Point out `JSONSerializer()`—this is why our `inference.py` yesterday needed `input_fn` and `output_fn`!
4. Execute the script. 
5. Emphasize the `update_endpoint` phase in the terminal output. Show how easy it is to change from 1 CPU instance to 3 instances seamlessly.
6. **Crucial:** Point out the `delete_endpoint()` call at the end. Stress that forgetting this line costs developers hundreds of dollars over the weekend.

## Summary
Reiterate that Cloud ML shifts the burden of physical server scaling directly onto AWS, allowing data scientists to deploy production APIs in a dozen lines of python.
