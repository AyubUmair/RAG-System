import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = str(PROJECT_ROOT / ".env"),
        env_file_encoding = 'utf-8',
        extra = 'ignore'
    )

    PROJECT_NAME : str = "Production Agentic RAG System"
    ENV : str = "development"
    LOG_LEVEL : str = "INFO"

    #Vector DB 
    QDRANT_HOST : str = "localhost"
    QDRANT_PORT : int = 6333
    QDRANT_GRPC_PORT : int = 6334
    QDRANT_COLLECTION_NAME : str = "agentic_rag_docs"
    QDRANT_API_KEY : Optional[str] = None
    QDRANT_URL : Optional[str] = None
    QDRANT_USE_LOCAL : bool = True
    QDRANT_LOCAL_PATH : str = str(Path(__file__).resolve().parent.parent / "data" / "qdrant_db")
    #Embedding and Sparse Model
    DENSE_EMBEDDING_MODEL : str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL : str = "Qdrant/bm25"
    EMBEDDING_DIM : int = 384

    #Reranker Model
    RERANKER_MODEL_NAME : str = "ms-marco-MiniLM-L-12-v2"
    TOP_K_HYBRID : int = 20
    TOP_K_RERANK : int = 6

    #LLM Settings
    LLM_PROVIDER : str = 'openai'
    OPENAI_API_KEY : Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL : str = "gpt-4o-mini"
    GEMINI_API_KEY : Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL : str = "gemini-3.6-flash"
    TEMPERATURE : float = 0.0
    
    #Web Search Fallback
    ENABLE_WEB_SEARCH_FALLBACK : bool = True
    TAVILY_API_KEY : Optional[str] = os.getenv("TAVILY_API_KEY")
    WEB_SEARCH_MAX_RESULTS : int = 4

    #Inestion and agent Guardrails
    CHUNK_SIZE : int = 800 
    CHUNK_OVERLAP : int = 150 
    MAX_RETRY_COUNT : int = 2
    RELEVANCE_THRESHOLD : float = 0.10

settings = Settings()