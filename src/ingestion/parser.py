import fitz
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharecterTextSpliter
from src.config import settings

class DocumentIngester:
    def __init__(self):
        self.text_splitter = RecursiveCharecterTextSpliter(
            chunk_size = settings.CHUNK_SIZE,
            chunk_overlap = settings.CHUNK_OVERLAP,
            seperators = ["\n\n", "\n", ".", " ", ""]
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