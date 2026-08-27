from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node
)
from src.config import settings

def decide_to_generate(state: AgentState) -> str:
    """
    Conditional Edge:
    - If documents are relevant -> go to generate.
    - If max retries reached -> go to generate.
    - Otherwise -> rewrite query and retry retrieval.
    """
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    
    if is_relevant or retry_count >= settings.MAX_RETRY_COUNT:
        return "generate"
    return "transform_query"

def create_agent_graph():
    
    builder = StateGraph(AgentState)
    
    # 1. Register Nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("generate", generate_node)
    
  
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")
    
    # 2. Add Conditional Routing
    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )
    
    # 4. Connect retry loop back to retrieve and exit on generate
    builder.add_edge("transform_query", "retrieve")
    builder.add_edge("generate", END)
    
    return builder.compile()


agent_app = create_agent_graph()