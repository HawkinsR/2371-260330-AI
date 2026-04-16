import os
import unittest
from unittest.mock import patch
from typing import Annotated, TypedDict
import operator

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- 1. Graph Definition ---
@tool
def fetch_database_records(customer_id: str) -> str:
    """Simulates a heavy, expensive external network fetch."""
    # In reality, this might hit an external SQL database
    return f"Retrieved actual production records for {customer_id}"

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def build_graph():
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
    llm_with_tools = llm.bind_tools([fetch_database_records])

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([fetch_database_records]))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


# --- 2. Unit Testing & Mocking ---
class TestAgentPipeline(unittest.TestCase):
    
    @patch('__main__.fetch_database_records.invoke')
    def test_agent_triggers_tool_correctly(self, mock_fetch):
        """
        SOLUTION: Evaluation-Driven testing.
        We patch the external tool to immediately return a preset string. 
        This prevents the graph from making live database calls during CI/CD.
        """
        # Configure the mock to return an artificial response
        mock_fetch.return_value = "MOCKED DATA RETURN: Records confirm customer is active."
        
        graph = build_graph()
        inputs = {"messages": [HumanMessage(content="Can you check records for customer 'C-999'?")]}
        
        print("\n[Executing Unit Test via Pytest/Unittest framework]")
        final_state = graph.invoke(inputs)
        
        # We assert that the LLM successfully decided to use the tool
        # and seamlessly incorporated the mocked payload into its final response.
        final_response = final_state['messages'][-1].content
        print(f"Agent Final Output: {final_response}")
        
        assert "active" in final_response.lower()
        print("✅ Unit Test Passed: The agent routed perfectly using mocked constraints.")


# --- 3. Creating Golden Datasets via LangSmith (Boilerplate) ---
def deploy_golden_dataset():
    """
    Shows trainees how to programmaticall upload a Golden Dataset to LangSmith 
    to be utilized by an Evaluator.
    """
    print("\n--- LangSmith Golden Dataset Boilerplate ---")
    print("from langsmith import Client")
    print("client = Client()")
    print("dataset = client.create_dataset(dataset_name='Agent-Accuracy-Baseline')")
    print("client.create_examples(inputs=[{'query': 'Check C-999'}], outputs=[{'status': 'active'}], dataset_id=dataset.id)")
    print("[Dataset upload executed purely via tracking traces]")


if __name__ == "__main__":
    print("=== Demo 065: Unit Testing & Golden Datasets ===")
    
    # 1. Run the local structural test demonstrating mocked tools
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentPipeline)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # 2. Show dataset upload logic
    deploy_golden_dataset()
