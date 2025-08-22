from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def process_document(self, document, metadata=None):
        """Process a document into chunks for embedding."""
        chunks = self.text_splitter.split_text(document)
        
        # Create metadata for each chunk
        if metadata:
            metadatas = [metadata.copy() for _ in range(len(chunks))]
            for i, meta in enumerate(metadatas):
                meta["chunk"] = i
                meta["chunk_total"] = len(chunks)
        else:
            metadatas = [{"chunk": i, "chunk_total": len(chunks)} for i in range(len(chunks))]
        
        return chunks, metadatas