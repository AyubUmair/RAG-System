from pathlib import Path
from src.config import settings

def test_settings_initialization():
    """Verify that settings load with valid types and correct default values."""
    assert settings.PROJECT_NAME == "Production Agentic RAG System"
    assert settings.EMBEDDING_DIM == 384
    assert settings.CHUNK_SIZE > settings.CHUNK_OVERLAP
    assert settings.RELEVANCE_THRESHOLD > 0.0

def test_qdrant_local_path_resolution():
    """Verify that QDRANT_LOCAL_PATH resolves to an absolute path inside data/qdrant_db."""
    db_path = Path(settings.QDRANT_LOCAL_PATH)
    
    # 1. Path must be absolute
    assert db_path.is_absolute(), f"Path is not absolute: {db_path}"
    
    # 2. Path must end with data/qdrant_db
    assert db_path.parts[-2:] == ("data", "qdrant_db"), f"Unexpected folder structure: {db_path}"
    
    # 3. Target parent directory ('data') must exist or be creatable
    db_path.parent.mkdir(parents=True, exist_ok=True)
    assert db_path.parent.exists(), f"Directory does not exist: {db_path.parent}"