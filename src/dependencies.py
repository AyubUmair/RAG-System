"""
Shared singleton instances. Import from here instead of instantiating
HybridVectorStore / Reranker directly anywhere else in the app — local-mode
Qdrant only allows one open client per storage folder, so creating a second
instance anywhere in the same process will crash with a file-lock error.
"""
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker

vector_store = HybridVectorStore()
reranker = Reranker()