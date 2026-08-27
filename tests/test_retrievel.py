from pathlib import Path
from qdrant_client import QdrantClient
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker

def test_hybrid_search_and_reranking(tmp_path: Path):
    """Test vector storage, hybrid retrieval, and cross-encoder reranking."""
    # Use an isolated in-memory Qdrant client for tests
    test_client = QdrantClient(location=":memory:")
    vector_store = HybridVectorStore(client=test_client, collection_name="test_collection")
    
    # Sample chunks with distinct topics
    sample_chunks = [
        {
            "text": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
            "metadata": {"source": "biology.txt", "page": 1, "chunk_id": "bio_1"}
        },
        {
            "text": "Vector databases store high-dimensional embeddings for fast approximate nearest neighbor search.",
            "metadata": {"source": "rag_guide.txt", "page": 1, "chunk_id": "rag_1"}
        },
        {
            "text": "LangGraph is a library for building stateful multi-actor applications with LLMs using graph workflows.",
            "metadata": {"source": "rag_guide.txt", "page": 2, "chunk_id": "rag_2"}
        }
    ]
    
    # 1. Upsert chunks
    vector_store.upsert_chunks(sample_chunks)
    
    # 2. Hybrid search for a database/RAG query
    query = "How do vector databases search embeddings?"
    retrieved_docs = vector_store.hybrid_search(query=query, limit=3)
    
    assert len(retrieved_docs) > 0
    # Top result should relate to vector databases
    assert "Vector databases" in retrieved_docs[0]["text"]
    
    # 3. Test Reranking
    reranker = Reranker()
    reranked_docs = reranker.rerank(
        query=query,
        documents=retrieved_docs,
        top_k=2,
        threshold=0.10
    )
    
    assert len(reranked_docs) >= 1
    assert "Vector databases" in reranked_docs[0]["text"]
    assert reranked_docs[0]["score"] > 0.10