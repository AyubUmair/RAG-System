import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import settings

class HybridVectorStore:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: str = settings.QDRANT_COLLECTION_NAME):
        
        self.collection_name = collection_name
        
        # 1. Initialize client 
        if client is not None:
            self.client = client
        elif settings.QDRANT_USE_LOCAL and not settings.QDRANT_URL:
            Path(settings.QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )

        
        self.dense_model = TextEmbedding(model_name=settings.DENSE_EMBEDDING_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL)

       
        self._ensure_collection()

    def _ensure_collection(self):
        
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={  
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
       
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        
      
        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        points = []
        for i, chunk in enumerate(chunks):
            sparse_vec = sparse_embeddings[i]
       
            raw_id = chunk.get("metadata", {}).get("chunk_id", str(uuid.uuid4()))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

            points.append(
                models.PointStruct(
                    id=point_id,
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
            collection_name=self.collection_name,
            points=points
        )

    def hybrid_search(self, query: str, limit: int = settings.TOP_K_HYBRID) -> List[Dict[str, Any]]:
        
        if not query.strip():
            return []

        query_dense = list(self.dense_model.embed([query]))[0].tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]
        
        results = self.client.query_points(
            collection_name=self.collection_name,
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
            {
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": point.score
            }
            for point in results.points
        ]