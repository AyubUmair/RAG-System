from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil

from src.ingestion.parser import DocumentIngester, UnsupportedFileTypeError, SUPPORTED_EXTENSIONS
from src.retrieval.vector_store import HybridVectorStore
from src.agent.graph import agent_app
from src.config import settings

app = FastAPI(title=settings.PROJECT_NAME)
ingester = DocumentIngester()
vector_store = HybridVectorStore()

# ... QueryRequest / QueryResponse unchanged ...

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
        parsed_docs = ingester.parse(temp_path)          # CHANGED: was ingester.parse_pdf(...)
        chunks = ingester.chunk_documents(parsed_docs)
        vector_store.upsert_chunks(chunks)
        return {"status": "success", "chunks_indexed": len(chunks), "file": file.filename}
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))