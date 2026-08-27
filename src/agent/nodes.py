from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker
from src.agent.state import AgentState


from langchain_tavily import TavilySearch
#from langchain_community.tools.tavily_search import TavilySearchResults

def _extract_text(content: Union[str, List[Any]]) -> str:
    """Helper to safely extract string text whether the LLM returns str or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)

def get_llm():
    """Factory to initialize the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        api_key = settings.GEMINI_API_KEY or "mock-key-for-tests"
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=settings.TEMPERATURE
        )
    
    api_key = settings.OPENAI_API_KEY or "mock-key-for-tests"
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=api_key,
        temperature=settings.TEMPERATURE
    )

vector_store = HybridVectorStore()
reranker = Reranker()

# --- NODE 1: RETRIEVAL ---
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves candidates via hybrid search and reranks them with FlashRank."""
    query = state.get("transformed_query") or state["question"]
    
    raw_docs = vector_store.hybrid_search(query=query, limit=settings.TOP_K_HYBRID)
    ranked_docs = reranker.rerank(
        query=query,
        documents=raw_docs,
        top_k=settings.TOP_K_RERANK,
        threshold=settings.RELEVANCE_THRESHOLD
    )
    
    return {
        "documents": ranked_docs,
        "source_documents": ranked_docs
    }

# --- NODE 2: DOCUMENT GRADER ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Relevance decision: 'yes' or 'no'")

def grade_documents_node(state: AgentState, llm_instance=None) -> Dict[str, Any]:
    """Determines whether the retrieved documents contain relevant context."""
    documents = state.get("documents", [])
    question = state["question"]
    
    if not documents:
        return {"is_relevant": False}

    active_llm = llm_instance or get_llm()
    structured_llm = active_llm.with_structured_output(GradeDocuments)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator assessing whether retrieved documents are relevant to a user question.\n"
                   "If the documents contain information that helps answer the question, grade as 'yes'.\n"
                   "Otherwise, grade as 'no'."),
        ("human", "User Question: {question}\n\nRetrieved Context:\n{context}")
    ])
    
    context_str = "\n\n".join([doc["text"] for doc in documents])
    response = (prompt | structured_llm).invoke({"question": question, "context": context_str})
    
    is_rel = response.binary_score.strip().lower() == "yes"
    return {"is_relevant": is_rel}

# --- NODE 3: QUERY REWRITER ---
def transform_query_node(state: AgentState, llm_instance=None) -> Dict[str, Any]:
    """Rewrites the query to improve retrieval recall on subsequent attempts."""
    base_question = state.get("transformed_query") or state["question"]  # CHANGED
    retry_count = state.get("retry_count", 0) + 1

    active_llm = llm_instance or get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query optimizer for search engines.\n"
                   "Analyze the user's initial question and produce an improved search query.\n"
                   "Focus on key technical nouns, concepts, and synonyms. Output ONLY the optimized search query."),
        ("human", "Initial Question: {question}")
    ])

    response = (prompt | active_llm).invoke({"question": base_question})  # CHANGED
    text_content = _extract_text(response.content)

    return {
        "transformed_query": text_content.strip(),
        "retry_count": retry_count
    }

# --- NODE 4: GENERATOR ---
def generate_node(state: AgentState, llm_instance=None) -> Dict[str, Any]:
    """Generates an answer strictly grounded in the retrieved documents."""
    question = state["question"]
    documents = state.get("documents", [])
    history = state.get("chat_history", [])

    if not documents:
        answer = "I could not find sufficient information in the provided documents to answer your question."
        return {
            "generation": answer,
            "chat_history": [                                    # NEW
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    context_str = "\n\n".join([
        f"[Source: {doc.get('metadata', {}).get('source', 'unknown')} | Page: {doc.get('metadata', {}).get('page', 1)}]\n{doc['text']}"
        for doc in documents
    ])

    history_str = "\n".join(                                      # NEW
        f"{turn['role'].upper()}: {turn['content']}" for turn in history[-6:]
    )

    active_llm = llm_instance or get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant answering user questions using the provided context.\n"
                   "Rules:\n"
                   "1. Answer strictly based on the provided context, not the conversation history.\n"
                   "2. Use the conversation history only to understand references and maintain continuity "
                   "   — never as a source of facts.\n"
                   "3. If the context does not contain the answer, say you don't know based on the documents.\n"
                   "4. Always cite sources inline, e.g., [Source: file.pdf | Page: 1]."),
        ("human", "Conversation history:\n{history}\n\nQuestion: {question}\n\nContext:\n{context}")  # CHANGED
    ])

    response = (prompt | active_llm).invoke({
        "history": history_str, "question": question, "context": context_str
    })
    text_content = _extract_text(response.content)

    return {
        "generation": text_content,
        "chat_history": [                                          # NEW
            {"role": "user", "content": question},
            {"role": "assistant", "content": text_content},
        ]
    }
# --- NODE 5: HALLUCINATION / GROUNDEDNESS GRADER ---
class GradeHallucination(BaseModel):
    """Binary score checking whether the generation is grounded in the retrieved context."""
    binary_score: str = Field(
        description="'yes' if every factual claim in the generation is supported by the "
                     "provided documents, 'no' if it contains unsupported claims or hallucinations."
    )
    reasoning: Optional[str] = Field(
        default=None, description="Brief justification for the score."
    )

def grade_generation_node(state: AgentState, llm_instance=None) -> Dict[str, Any]:
    """Checks whether the generated answer is factually grounded in the retrieved documents."""
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    # Nothing was retrieved (e.g. "I could not find sufficient information..." fallback) —
    # there's nothing to hallucinate against, so treat as grounded and let it pass through.
    if not documents or not generation.strip():
        return {"is_grounded": True}

    active_llm = llm_instance or get_llm()
    structured_llm = active_llm.with_structured_output(GradeHallucination)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict fact-checker. Compare the GENERATED ANSWER against the SOURCE "
         "DOCUMENTS.\n"
         "Grade 'yes' only if every factual claim in the answer is directly supported by the "
         "documents.\n"
         "Grade 'no' if the answer contains any claim, number, name, or detail that is not "
         "present in the documents, even if it seems plausible or generally true.\n"
         "An answer that correctly says 'the documents don't contain this information' should "
         "be graded 'yes'."),
        ("human", "SOURCE DOCUMENTS:\n{context}\n\nGENERATED ANSWER:\n{generation}")
    ])

    context_str = "\n\n".join([doc["text"] for doc in documents])
    response = (prompt | structured_llm).invoke({
        "context": context_str,
        "generation": generation
    })

    is_grounded = response.binary_score.strip().lower() == "yes"
    return {"is_grounded": is_grounded}
# --- NODE: WEB SEARCH FALLBACK ---
def web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Falls back to live web search when local retrieval + retries produced nothing relevant.
    Formats results into the same document schema used by the vector store so downstream
    nodes (generate, grade_generation) don't need to know the source.
    """
    if not settings.ENABLE_WEB_SEARCH_FALLBACK or not settings.TAVILY_API_KEY:
        # No fallback configured — pass through with whatever (possibly empty) documents exist.
        return {"used_web_search": False}

    query = state.get("transformed_query") or state["question"]

# in web_search_node
    search_tool = TavilySearch(
    api_key=settings.TAVILY_API_KEY,
    max_results=settings.WEB_SEARCH_MAX_RESULTS
    )

    try:
        response = search_tool.invoke({"query": query})
        raw_results = response.get("results", []) if isinstance(response, dict) else response
    except Exception:
        return {"used_web_search": False}
    web_docs = [
        {
            "text": r.get("content", ""),
            "metadata": {
                "source": r.get("url", "web"),
                "page": None,
                "title": r.get("title", "Web result")
            },
            "score": None
        }
        for r in raw_results
        if r.get("content", "").strip()
    ]

    return {
        "documents": web_docs,
        "source_documents": web_docs,
        "used_web_search": True
    }
# --- NODE: CONTEXTUALIZE QUERY (resolves follow-ups using chat history) ---
def contextualize_query_node(state: AgentState, llm_instance=None) -> Dict[str, Any]:
    """
    Rewrites a follow-up question ("what about page 3?") into a standalone question
    using prior turns, so retrieval doesn't depend on conversational context it can't see.
    Runs once at the start of every turn, before retrieval.
    """
    history = state.get("chat_history", [])
    question = state["question"]

    if not history:
        return {"transformed_query": None}  # first turn — nothing to resolve against

    active_llm = llm_instance or get_llm()
    history_str = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}" for turn in history[-6:]  # last ~3 exchanges
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and a follow-up question, rewrite the follow-up into a "
         "standalone question containing all context needed to understand it alone. Do NOT answer "
         "it. If it's already standalone, return it unchanged. Output ONLY the rewritten question."),
        ("human", "Conversation history:\n{history}\n\nFollow-up question: {question}")
    ])

    response = (prompt | active_llm).invoke({"history": history_str, "question": question})
    return {"transformed_query": _extract_text(response.content).strip()}

