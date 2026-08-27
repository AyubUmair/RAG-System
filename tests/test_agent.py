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