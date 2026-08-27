import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent.state import AgentState
from src.agent.nodes import (
    contextualize_query_node,   # NEW
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    grade_generation_node,
    web_search_node,
)
from src.config import settings

def decide_to_generate(state: AgentState) -> str:
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    if is_relevant:
        return "generate"
    if retry_count >= settings.MAX_RETRY_COUNT:
        return "web_search"
    return "transform_query"

def decide_after_generation(state: AgentState) -> str:
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    used_web_search = state.get("used_web_search", False)
    if is_grounded or retry_count >= settings.MAX_RETRY_COUNT or used_web_search:
        return "end"
    return "transform_query"

def create_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("contextualize", contextualize_query_node)   # NEW
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate_node)
    builder.add_node("grade_generation", grade_generation_node)

    builder.set_entry_point("contextualize")                       # CHANGED (was "retrieve")
    builder.add_edge("contextualize", "retrieve")                   # NEW
    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"}
    )

    builder.add_edge("transform_query", "retrieve")
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", "grade_generation")

    builder.add_conditional_edges(
        "grade_generation",
        decide_after_generation,
        {"end": END, "transform_query": "transform_query"}
    )

    # --- Persistent checkpointing for multi-turn sessions ---
    checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "data"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_dir / "checkpoints.sqlite"), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)

agent_app = create_agent_graph()