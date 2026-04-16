import json
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

# TODO: 3. Update the agent node to use ChatBedrock instead of this static mock.
def agent_node(state):
    return {"messages": ["Welcome to the LangGraph Deployment Task!"]}

builder = StateGraph(dict)
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
