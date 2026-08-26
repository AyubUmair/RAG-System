from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List, Dict, Any
from src.config import settings

class HybridVectorStore:
    def __init__(self):
        if settings.QDRANT_USE_LOCAL and not settings.QDRANT_URL :
            self.client = QdrantClient(path = settings.QDRANT_LOCAL_PATH)
        else:
            self.client = QdrantClient(
                host = settings.QDRANT_HOST,
                port = settings.QDRANT_PORT,
                api_key = settings.QDRANT_API_KEY
            )

        self.dense_model = TextEmbedding(model_name = settings.DENSE_EMBEDDING_MODEL)
        self.sparse_moel = SparseTextEmbedding(model_name = settings.SPARSE_EMBEDDING_MODEL)

    def _ensure_collction(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if settings.QDRANT_COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name = settings.QDRANT_COLLECTION_NAME,
                vectors_config = {
                    'dense': models.VectorParams(
                        size = settings.EMBEDDING_DIM,
                        distance = models.Distance.COSINE
                    )
                },
                sparse_vector_config = {
                    'sparse' : models.SparseVectorParams(
                        index = models.SparseIndexParams(on_disk = False)
                    )
                }
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates dense + sparse embeddings and stores points in Qdrant."""
        texts = [c["text"] for c in chunks]
        
        # FastEmbed batch embeddings
        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        points = []
        for i, chunk in enumerate(chunks):
            sparse_vec = sparse_embeddings[i]
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist()
                        )
                    },
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                )
            )
        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
    def hybrid_search(self, query: str, limit: int = settings.TOP_K_HYBRID) -> List[Dict[str, Any]]:
        """Executes Reciprocal Rank Fusion (RRF) between dense and sparse results."""
        query_dense = list(self.dense_model.embed([query]))[0].tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]
        
        # Qdrant query API performing RRF fusion over dense and sparse vectors
        results = self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        
        return [
            {"text": point.payload["text"], "metadata": point.payload["metadata"], "score": point.score}
            for point in results.points
        ]
    