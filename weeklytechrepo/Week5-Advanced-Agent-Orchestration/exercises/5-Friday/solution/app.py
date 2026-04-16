import json
import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

# SOLUTION 3: Authentic Bedrock Implementation
def agent_node(state: AgentState):
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
    response = llm.invoke(state["messages"][-1])
    return {"messages": [response.content]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_input = body.get("message", "Test")
        final_state = graph.invoke({"messages": [user_input]})
        
        return {
            "statusCode": 200,
            "body": json.dumps({"response": final_state["messages"][-1]})
        }
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}
