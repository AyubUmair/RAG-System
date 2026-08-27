from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    grade_generation_node,   # NEW
)
from src.config import settings

def decide_to_generate(state: AgentState) -> str:
    """
    Conditional Edge (after document grading):
    - If documents are relevant -> go to generate.
    - If max retries reached -> go to generate anyway.
    - Otherwise -> rewrite query and retry retrieval.
    """
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)

    if is_relevant or retry_count >= settings.MAX_RETRY_COUNT:
        return "generate"
    return "transform_query"

def decide_after_generation(state: AgentState) -> str:
    """
    Conditional Edge (after groundedness grading):
    - If the generation is grounded -> done.
    - If not grounded but retries remain -> rewrite query and retry the whole loop.
    - If not grounded and retries are exhausted -> stop anyway (avoid infinite loop),
      but the caller can inspect `is_grounded` to warn the user.
    """
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)

    if is_grounded or retry_count >= settings.MAX_RETRY_COUNT:
        return "end"
    return "transform_query"

def create_agent_graph():

    builder = StateGraph(AgentState)

    # 1. Register Nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("generate", generate_node)
    builder.add_node("grade_generation", grade_generation_node)  # NEW

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # 2. Document relevance routing
    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )

    # 3. Retry loop back to retrieve
    builder.add_edge("transform_query", "retrieve")

    # 4. Generation now flows into groundedness check instead of straight to END
    builder.add_edge("generate", "grade_generation")

    # 5. Groundedness routing — loop back on hallucination, otherwise finish
    builder.add_conditional_edges(
        "grade_generation",
        decide_after_generation,
        {
            "end": END,
            "transform_query": "transform_query"
        }
    )

    return builder.compile()


agent_app = create_agent_graph()