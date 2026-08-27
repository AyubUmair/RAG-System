from src.agent.graph import decide_to_generate, decide_after_generation
from src.agent.state import AgentState


# --- decide_to_generate (routing after document grading) ---

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
    """If retry limit reached and docs still irrelevant, fall back to web search instead of giving up."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": "Optimized query",
        "documents": [],
        "generation": None,
        "retry_count": 2,  # Max retries reached
        "is_relevant": False,
        "source_documents": []
    }
    assert decide_to_generate(state) == "web_search"


def test_decide_to_generate_when_retry_limit_reached_but_relevant():
    """If documents are relevant, go straight to generate regardless of retry count."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": "Optimized query",
        "documents": [{"text": "Agentic RAG explanation", "metadata": {}}],
        "generation": None,
        "retry_count": 2,
        "is_relevant": True,
        "source_documents": []
    }
    assert decide_to_generate(state) == "generate"


# --- decide_after_generation (routing after groundedness grading) ---

def test_decide_after_generation_when_grounded():
    """If the generation is grounded, the graph should end."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": None,
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG combines retrieval with agent decision-making.",
        "retry_count": 0,
        "is_relevant": True,
        "is_grounded": True,
        "used_web_search": False,
        "source_documents": []
    }
    assert decide_after_generation(state) == "end"


def test_decide_after_generation_when_hallucinated_and_retries_remain():
    """If ungrounded and retries remain, loop back to transform_query."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": None,
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG was invented in 1998 by John Smith.",
        "retry_count": 0,
        "is_relevant": True,
        "is_grounded": False,
        "used_web_search": False,
        "source_documents": []
    }
    assert decide_after_generation(state) == "transform_query"


def test_decide_after_generation_when_retries_exhausted():
    """If ungrounded but out of retries, stop anyway to avoid infinite loop."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": None,
        "documents": [{"text": "...", "metadata": {}}],
        "generation": "Agentic RAG was invented in 1998 by John Smith.",
        "retry_count": 2,
        "is_relevant": True,
        "is_grounded": False,
        "used_web_search": False,
        "source_documents": []
    }
    assert decide_after_generation(state) == "end"


def test_decide_after_generation_when_ungrounded_after_web_search():
    """If the answer came from web search and still isn't grounded, stop rather than loop again."""
    state: AgentState = {
        "question": "What is Agentic RAG?",
        "transformed_query": "Optimized query",
        "documents": [{"text": "some web result", "metadata": {"source": "web"}}],
        "generation": "Agentic RAG was invented in 1998 by John Smith.",
        "retry_count": 2,
        "is_relevant": False,
        "is_grounded": False,
        "used_web_search": True,
        "source_documents": []
    }
    assert decide_after_generation(state) == "end"