from src.agent.graph import decide_to_generate
from src.agent.state import AgentState

def test_decide_to_generate_when_relevant():
    """If documents are graded as relevant, route directly to generate."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": None,
        "documents": [{"text": "Agentic RAG explanation", "metadata": {}}],
        "generation": None,
        "retry_count": 0,
        "is_relevant": True,
        "source_documents": []
    }
    assert decide_to_generate(state) == "generate"

def test_decide_to_generate_when_irrelevant_and_under_retry_limit():
    """If documents are irrelevant and retries remaining, route to transform_query."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": None,
        "documents": [],
        "generation": None,
        "retry_count": 0,
        "is_relevant": False,
        "source_documents": []
    }
    assert decide_to_generate(state) == "transform_query"

def test_decide_to_generate_when_retry_limit_reached():
    """If retry limit reached, force progression to generate even if not relevant."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": "Optimized query",
        "documents": [],
        "generation": None,
        "retry_count": 2,  # Max retries reached
        "is_relevant": False,
        "source_documents": []
    }
    assert decide_to_generate(state) == "generate"

from src.agent.graph import decide_after_generation

def test_decide_after_generation_when_grounded():
    """If the generation is grounded, the graph should end."""
    state = {
        "question": "What is Agentic RAG?",
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG combines retrieval with agent decision-making.",
        "retry_count": 0,
        "is_grounded": True,
    }
    assert decide_after_generation(state) == "end"

def test_decide_after_generation_when_hallucinated_and_retries_remain():
    """If ungrounded and retries remain, loop back to transform_query."""
    state = {
        "question": "What is Agentic RAG?",
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG was invented in 1998 by John Smith.",
        "retry_count": 0,
        "is_grounded": False,
    }
    assert decide_after_generation(state) == "transform_query"

def test_decide_after_generation_when_retries_exhausted():
    """If ungrounded but out of retries, stop anyway to avoid infinite loop."""
    state = {
        "question": "What is Agentic RAG?",
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG was invented in 1998 by John Smith.",
        "retry_count": 2,
        "is_grounded": False,
    }
    assert decide_after_generation(state) == "end"