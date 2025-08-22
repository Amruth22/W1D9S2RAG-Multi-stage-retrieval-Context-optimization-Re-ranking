import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.services.embedding import EmbeddingService
import os
import uuid

class ChromaService:
    def __init__(self, collection_name="documents"):
        # Ensure directory exists
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        self.embedding_service = EmbeddingService()
    
    async def add_documents(self, documents, metadatas=None):
        """Add documents to the vector store."""
        if not documents:
            return []
        
        # Generate IDs if not provided
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_service.get_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas if metadatas else [{"source": "user_upload"} for _ in range(len(documents))]
        )
        
        return ids
    
    async def similarity_search(self, query, n_results=5, **kwargs):
        """Search for similar documents to the query."""
        query_embedding = self.embedding_service.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            **kwargs
        )
        
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }