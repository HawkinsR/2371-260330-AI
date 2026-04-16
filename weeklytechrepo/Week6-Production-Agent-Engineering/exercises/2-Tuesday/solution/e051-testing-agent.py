import unittest
from unittest.mock import patch
from typing import Annotated, TypedDict
import operator

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def query_financial_database(ticker: str) -> str:
    """Queries a secure financial database. WARNING: Costs $1 per API execution."""
    import time
    time.sleep(3)
    return "Error: Database timeout."

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def build_graph():
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
    llm_with_tools = llm.bind_tools([query_financial_database])

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([query_financial_database]))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()

class TestAgentFeatures(unittest.TestCase):
    
    # SOLUTION 1: Override external API hit
    @patch('__main__.query_financial_database.invoke')
    def test_financial_routing(self, mock_query): # SOLUTION 2: parameter injection
        
        # SOLUTION 3: set intercept value
        mock_query.return_value = "TESTING_FLAG: Revenue is 10 million dollars."
        
        graph = build_graph()
        inputs = {"messages": [HumanMessage(content="What is the revenue for TSLA?")]}
        
        print("\n--- Executing Graph Offline ---")
        final_state = graph.invoke(inputs)
        result = final_state["messages"][-1].content
        
        print(f"Agent Final Print: {result}")
        
        # SOLUTION 4: Assert agent successfully processed the mock
        self.assertIn("10 million", result.lower())
        print("✅ Success: Graph successfully executed local mock without timing out.")

if __name__ == "__main__":
    unittest.main()
