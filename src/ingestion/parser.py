import fitz
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from src.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension isn't in SUPPORTED_EXTENSIONS."""
    pass


class DocumentIngester:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # --- DISPATCHER ---
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Routes a file to the correct parser based on its extension.
        Every parser returns the same schema: [{"text": ..., "metadata": {...}}, ...]
        """
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".pdf":
            return self.parse_pdf(file_path)
        if suffix == ".docx":
            return self.parse_docx(file_path)
        if suffix in (".html", ".htm"):
            return self.parse_html(file_path)
        if suffix in (".md", ".markdown"):
            return self.parse_markdown(file_path)
        if suffix == ".txt":
            return self.parse_plain_text_file(file_path)

    # --- PDF (unchanged) ---
    def parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        documents = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_num + 1,
                        "total_page": len(doc)
                    }
                })
        doc.close()
        return documents

    # --- DOCX ---
    def parse_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Uses unstructured's docx partitioner, which segments by element (heading,
        paragraph, table, list item) rather than raw page breaks — DOCX has no
        native concept of "pages" the way PDF does.
        """
        elements = partition_docx(filename=str(file_path))
        return self._elements_to_documents(elements, file_path.name)

    # --- HTML ---
    def parse_html(self, file_path: Path) -> List[Dict[str, Any]]:
        """Strips tags/nav/boilerplate and extracts the meaningful content blocks."""
        elements = partition_html(filename=str(file_path))
        return self._elements_to_documents(elements, file_path.name)

    # --- MARKDOWN ---
    def parse_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses markdown structure (headers, lists, code blocks) into elements."""
        elements = partition_md(filename=str(file_path))
        return self._elements_to_documents(elements, file_path.name)

    # --- PLAIN TEXT FILE ---
    def parse_plain_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        elements = partition_text(filename=str(file_path))
        return self._elements_to_documents(elements, file_path.name)

    # --- RAW STRING (kept for backward compatibility / programmatic use) ---
    def parse_text(self, text: str, source_name: str = "raw_text") -> List[Dict[str, Any]]:
        """Parses a raw in-memory string into a standard document format (no file on disk)."""
        if not text.strip():
            return []
        return [{
            "text": text.strip(),
            "metadata": {
                "source": source_name,
                "page": 1,
                "total_pages": 1
            }
        }]

    # --- SHARED HELPER for unstructured-based parsers ---
    def _elements_to_documents(self, elements, source_name: str) -> List[Dict[str, Any]]:
        """
        Groups unstructured 'elements' into one document per section, using each
        element's own page_number metadata when the format provides one (DOCX/PDF-like),
        and falling back to page=1 for formats without real pagination (HTML/MD/TXT).
        """
        documents = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            page_number = getattr(el.metadata, "page_number", None) or 1
            documents.append({
                "text": text,
                "metadata": {
                    "source": source_name,
                    "page": page_number,
                    "element_type": type(el).__name__,  # e.g. "Title", "NarrativeText", "Table"
                }
            })
        return documents

    # --- CHUNKING (unchanged) ---
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            for idx, chunk_text in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": f"{doc['metadata']['source']}_p{doc['metadata']['page']}_c{idx}"
                    }
                })
        return chunked_docs