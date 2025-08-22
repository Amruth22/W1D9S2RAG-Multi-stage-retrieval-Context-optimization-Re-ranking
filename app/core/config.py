import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    API_TITLE: str = "Multi-Stage RAG API"
    API_DESCRIPTION: str = "RAG system with multi-stage retrieval and context optimization"
    API_VERSION: str = "0.1.0"
    
    # Gemini API
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-pro"
    
    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
    
    # Retrieval settings
    MAX_CHUNKS_FIRST_STAGE: int = 10
    MAX_CHUNKS_RETURNED: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

settings = Settings()