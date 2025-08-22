from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def get_embedding(self, text):
        """Generate embedding for a single text."""
        if not text:
            return []
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()