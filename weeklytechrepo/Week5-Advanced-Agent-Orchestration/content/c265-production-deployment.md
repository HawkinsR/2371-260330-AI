# c265: Production Deployment

## Deployment Options
Once your LangGraph workflow is functionally complete, moving from a local script to a production environment is the final step. Depending on enterprise requirements, deploying a multi-agent system requires scalable infrastructure capable of managing asynchronous threads, checkpointer persistence, and concurrent graph executions.

### LangGraph Platform Overview
The official ecosystem offers two main deployment strategies:
- **LangGraph Cloud**: A fully managed SaaS platform designed specifically for hosting LangGraph architectures. It seamlessly handles threaded checkpoints, memory scaling, and streaming API responses out of the box.
- **Self-Hosted (LangGraph Platform)**: For organizations with strict data residency or VPC requirements, the LangGraph Platform can be deployed via Docker containers onto existing infrastructure like AWS ECS, GCP Cloud Run, or Azure App Service. This approach still provides the LangGraph Studio interface and API server, but runs entirely within your perimeter.
