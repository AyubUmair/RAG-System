import fitz
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class DocumentIngester:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Note: 'separators' spelled with an 'a'
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        documents = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                documents.append({
                    "text" : text,
                    "metadata" : {
                        "source" : file_path.name,
                        "page" : page_num +1 ,
                        "total_page" : len(doc)
                    }
                }

                )
        doc.close()
        return documents
    
    def parse_text(self, text: str, source_name: str = "raw_text") -> List[Dict[str, Any]]:
        """Parses plain text strings into a standard document format."""
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
    
    def chunk_documents(self, documents : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            for idx, chunk_text in enumerate(chunks):
                chunked_docs.append({
                    "text" : chunk_text , 
                    "metadata" : {
                        **doc["metadata"],
                        "chunk_id" : f"{doc['metadata']['source']}_p{doc['metadata']['page']}_c{idx}"
                    }
                }

                )
        return chunked_docs