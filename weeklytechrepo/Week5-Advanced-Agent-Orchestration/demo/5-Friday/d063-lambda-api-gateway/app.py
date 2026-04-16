import json
import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# 1. State
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

# 2. Graph Definition
def agent_node(state: AgentState):
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
    prompt = f"You are a deployed LangGraph agent in AWS Lambda. Answer concisely: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [response.content]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()

# 3. AWS Lambda Handler
def lambda_handler(event, context):
    """
    Standard AWS Lambda entry point triggered by API Gateway.
    """
    try:
        print(f"Received Event: {event}")
        
        # Parse the incoming HTTP POST request from API Gateway
        body = json.loads(event.get("body", "{}"))
        user_input = body.get("message", "Hello")
        
        # Invoke the LangGraph workflow
        final_state = graph.invoke({"messages": [user_input]})
        
        # Extract the final LLM response from the state list
        agent_response = final_state["messages"][-1]
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "source": "LangGraph Lambda",
                "response": agent_response
            })
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
