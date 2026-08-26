from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str                   # Original user query
    transformed_query: Optional[str]# Rewritten query for retry
    documents: List[Dict[str, Any]] # Retrieved & filtered context chunks
    generation: Optional[str]       # Generated answer
    retry_count: int                # Number of re-retrieval attempts
    is_relevant: bool               # Document quality decision flag
    source_documents: List[Dict[str, Any]] # Final citations