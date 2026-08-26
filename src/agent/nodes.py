from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker
from src.agent.state import AgentState

# Select LLM
def get_llm():
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.TEMPERATURE
        )
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=settings.TEMPERATURE
    )

vector_store = HybridVectorStore()
reranker = Reranker()
llm = get_llm()

# --- NODE 1: RETRIEVAL ---
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves candidates using hybrid search and applies cross-encoder reranking."""
    query = state.get("transformed_query") or state["question"]
    raw_docs = vector_store.hybrid_search(query=query)
    ranked_docs = reranker.rerank(query=query, documents=raw_docs)
    
    return {
        "documents": ranked_docs,
        "source_documents": ranked_docs
    }

# --- NODE 2: DOCUMENT GRADER ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """Determines whether the retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"is_relevant": False}

    structured_llm = llm.with_structured_output(GradeDocuments)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator assessing relevance of retrieved documents to a user query.\n"
                   "If the document contains keywords or semantic meaning related to the user question, grade it as 'yes'.\n"
                   "Give a binary score 'yes' or 'no'."),
        ("human", "User question: {question}\n\nRetrieved Context:\n{context}")
    ])
    
    context_str = "\n\n".join([doc["text"] for doc in documents])
    response = (prompt | structured_llm).invoke({"question": question, "context": context_str})
    
    is_rel = (response.binary_score.strip().lower() == "yes")
    return {"is_relevant": is_rel}

# --- NODE 3: QUERY REWRITER ---
def transform_query_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites the query to optimize for vector/BM25 retrieval."""
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query optimizer for search engines.\n"
                   "Analyze the original question and formulate an improved keyword and semantic search query "
                   "that resolves ambiguities and maximizes retrieval recall."),
        ("human", "Original Question: {question}\n\nProvide ONLY the optimized query string.")
    ])
    
    response = (prompt | llm).invoke({"question": question})
    return {
        "transformed_query": response.content.strip(),
        "retry_count": retry_count
    }

# --- NODE 4: GENERATION ---
def generate_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the answer grounded strictly in retrieved documents."""
    question = state["question"]
    documents = state.get("documents", [])
    
    context_str = "\n\n".join([
        f"[Source: {doc['metadata'].get('source', 'unknown')} | Page: {doc['metadata'].get('page', 1)}]\n{doc['text']}"
        for doc in documents
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant answering questions using provided context documents.\n"
                   "Rules:\n"
                   "1. Ground your answer strictly in the context.\n"
                   "2. If the context does not contain enough information, state clearly that the answer is not in the documents.\n"
                   "3. Always cite sources inline like [Source: filename, Page: X]."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    response = (prompt | llm).invoke({"question": question, "context": context_str})
    return {"generation": response.content}