from pathlib import Path
import shutil
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.ingestion.parser import DocumentIngester
from src.agent.nodes import vector_store
from src.agent.graph import agent_app

app = FastAPI(title=settings.PROJECT_NAME)
ingester = DocumentIngester()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retry_count: int

@app.get("/health")
async def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Uploads a PDF, chunks it, and indexes it into Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_dir = Path("data/raw")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        parsed_pages = ingester.parse_pdf(temp_path)
        chunks = ingester.chunk_documents(parsed_pages)
        vector_store.upsert_chunks(chunks)
        return {
            "status": "success",
            "filename": file.filename,
            "pages_parsed": len(parsed_pages),
            "chunks_indexed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Executes the LangGraph Agentic RAG workflow."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial_state = {
        "question": request.question.strip(),
        "transformed_query": None,
        "documents": [],
        "generation": None,
        "retry_count": 0,
        "is_relevant": False,
        "source_documents": []
    }

    try:
        final_state = agent_app.invoke(initial_state)
        
        raw_answer = final_state.get("generation", "No response generated.")
        if isinstance(raw_answer, list):
            answer_str = "".join([
                p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_answer
            ])
        else:
            answer_str = str(raw_answer)

        return QueryResponse(
            answer=answer_str,
            sources=final_state.get("source_documents", []),
            grounded=final_state.get("is_grounded", True),
            retry_count=final_state.get("retry_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")