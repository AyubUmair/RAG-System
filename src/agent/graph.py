from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    grade_generation_node,
    web_search_node,          # NEW
)
from src.config import settings

def decide_to_generate(state: AgentState) -> str:
    """
    Conditional Edge (after document grading):
    - Relevant docs -> generate.
    - Not relevant, retries remain -> rewrite query and retry retrieval.
    - Not relevant, retries exhausted -> fall back to web search instead of giving up.
    """
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)

    if is_relevant:
        return "generate"
    if retry_count >= settings.MAX_RETRY_COUNT:
        return "web_search"       # CHANGED: was "generate"
    return "transform_query"

def decide_after_generation(state: AgentState) -> str:
    """
    Conditional Edge (after groundedness grading).
    Note: if the answer came from web_search and still isn't grounded, we stop rather than
    loop again — a second web search round rarely fixes a groundedness failure and risks
    an expensive/slow loop.
    """
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    used_web_search = state.get("used_web_search", False)

    if is_grounded or retry_count >= settings.MAX_RETRY_COUNT or used_web_search:
        return "end"
    return "transform_query"

def create_agent_graph():

    builder = StateGraph(AgentState)

    # 1. Register Nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("web_search", web_search_node)         # NEW
    builder.add_node("generate", generate_node)
    builder.add_node("grade_generation", grade_generation_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # 2. Document relevance routing (now can go to web_search too)
    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",                     # NEW
        }
    )

    # 3. Retry loop back to retrieve
    builder.add_edge("transform_query", "retrieve")

    # 4. Web search results flow straight into generation
    builder.add_edge("web_search", "generate")               # NEW

    # 5. Generation -> groundedness check
    builder.add_edge("generate", "grade_generation")

    # 6. Groundedness routing
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