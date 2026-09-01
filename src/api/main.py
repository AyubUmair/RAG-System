import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import traceback


from src.ingestion.parser import DocumentIngester, UnsupportedFileTypeError, SUPPORTED_EXTENSIONS
from src.dependencies import vector_store
from src.agent.graph import agent_app
from src.config import settings

app = FastAPI(title=settings.PROJECT_NAME)
ingester = DocumentIngester()


class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    retry_count: int
    grounded: bool
    used_web_search: bool
    thread_id: str


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Uploads a document (PDF, DOCX, HTML, Markdown, or TXT), chunks it, and indexes into Qdrant."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    temp_dir = Path("data/raw")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        parsed_docs = ingester.parse(temp_path)
        chunks = ingester.chunk_documents(parsed_docs)
        vector_store.upsert_chunks(chunks)
        return {"status": "success", "chunks_indexed": len(chunks), "file": file.filename}
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Invokes the LangGraph CRAG workflow, persisting state per thread_id."""
    thread_id = request.thread_id or str(uuid.uuid4())

    initial_state = {
        "question": request.question,
        "transformed_query": None,
        "documents": [],
        "generation": None,
        "retry_count": 0,
        "is_relevant": False,
        "is_grounded": True,
        "used_web_search": False,
        "source_documents": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = agent_app.invoke(initial_state, config=config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

    return QueryResponse(
        answer=final_state.get("generation", "No response generated."),
        sources=final_state.get("source_documents", []),
        retry_count=final_state.get("retry_count", 0),
        grounded=final_state.get("is_grounded", True),
        used_web_search=final_state.get("used_web_search", False),
        thread_id=thread_id,
    )

    