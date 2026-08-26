from flashrank import Ranker, RerankRequest
from typing import List, Dict, Any
from src.config import settings

class Reranker:
    def __init__(self):
        # Ultra-fast, lightweight quantized reranker running locally
        self.ranker = Ranker(model_name=settings.RERANKER_MODEL_NAME, cache_dir="/tmp/flashrank")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = settings.TOP_K_RERANK) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        passages = [
            {"id": idx, "text": doc["text"], "metadata": doc["metadata"]}
            for idx, doc in enumerate(documents)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)
        
        # Filter by top_k and threshold
        filtered = []
        for res in ranked_results[:top_k]:
            if res["score"] >= settings.RELEVANCE_THRESHOLD:
                filtered.append({
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "score": float(res["score"])
                })
        return filtered