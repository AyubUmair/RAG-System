from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from src.config import settings

class Reranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = settings.TOP_K_RERANK,
        threshold: float = settings.RELEVANCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
       
        if not documents or not query.strip():
            return []
        
        passages = [
            {"id": idx, "text": doc["text"], "metadata": doc.get("metadata", {})}
            for idx, doc in enumerate(documents)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)
        
        # Filter top-k by relevance threshold
        filtered = []
        for res in ranked_results[:top_k]:
            score = float(res["score"])
            if score >= threshold:
                filtered.append({
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "score": score
                })
        return filtered