# Multi-Stage Retrieval RAG System - Project Description

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system with multi-stage retrieval, context optimization, and preparation for re-ranking. The system enhances traditional RAG approaches by implementing a sophisticated four-stage retrieval pipeline that improves the quality and relevance of retrieved passages before generating responses.

## Key Features

1. **Multi-stage retrieval** - A three-stage retrieval pipeline that improves result quality
2. **Context optimization** - Smart techniques to organize and prioritize retrieved passages
3. **PDF processing** - Built-in capabilities to extract and process text from PDF documents
4. **FastAPI interface** - Clean API endpoints with Swagger documentation
5. **Query expansion** - Uses Gemini API to rewrite and expand queries for better recall
6. **Re-ranking with FlashRank** - Custom implementation of a feature-based re-ranking system

## Tech Stack

- **Python 3.9+** - Core programming language
- **FastAPI** - API framework with automatic Swagger documentation
- **BAAI/bge-small-en-v1.5** - Embedding model for text vectorization
- **ChromaDB** - Vector database for similarity search
- **Google Generative AI** - Gemini API for query expansion and response generation
- **PyPDF2 & PDFPlumber** - PDF text extraction libraries
- **LangChain** - Text chunking and document processing tools
- **Uvicorn** - ASGI web server for FastAPI

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── models.py          # Pydantic models for API validation
│   │   └── routes.py          # API endpoint definitions
│   ├── core/
│   │   └── config.py          # Configuration settings
│   ├── services/
│   │   ├── document_processor.py  # Text document chunking
│   │   ├── embedding.py       # Embedding generation service
│   │   ├── gemini.py          # Gemini API integration
│   │   ├── pdf_processor.py   # PDF text extraction
│   │   ├── ranker.py          # Custom FlashRank implementation
│   │   ├── retriever.py       # Multi-stage retrieval logic
│   │   └── vector_store.py    # ChromaDB integration
│   ├── main.py                # FastAPI application setup
│   └── __init__.py
├── data/
│   └── chroma/               # Persistent ChromaDB storage
├── .env                      # Environment variables
├── .gitignore
├── example_usage.py          # Example client implementation
├── main.py                   # Application entry point
├── README.md
└── requirements.txt
```

## Core Components

### 1. API Layer (`app/api/`)

#### Models (`app/api/models.py`)
- `DocumentIn` - Input model for text documents
- `DocumentsIn` - Wrapper for multiple documents
- `QueryIn` - Input model for queries
- `DocumentOut` - Output model for retrieved documents
- `QueryResult` - Response model for query results
- `StatusResponse` - Generic status response

#### Routes (`app/api/routes.py`)
- `POST /api/documents` - Add text documents to the system
- `POST /api/upload-pdf` - Upload and process PDF documents
- `POST /api/query` - Query the RAG system with multi-stage retrieval

### 2. Services Layer (`app/services/`)

#### Document Processing (`app/services/document_processor.py`)
- Uses LangChain's RecursiveCharacterTextSplitter
- Splits documents into chunks with configurable size and overlap
- Maintains metadata for each chunk

#### PDF Processing (`app/services/pdf_processor.py`)
- Dual extraction approach using PyPDF2 and pdfplumber
- Falls back to pdfplumber if PyPDF2 extraction is insufficient
- Integrates with document processor for chunking

#### Embedding Service (`app/services/embedding.py`)
- Uses sentence-transformers with BAAI/bge-small-en-v1.5 model
- Provides both single and batch embedding generation
- Normalizes embeddings for consistency

#### Vector Store (`app/services/vector_store.py`)
- ChromaDB integration with persistent storage
- Handles document addition with embeddings
- Performs similarity search with configurable result count

#### Gemini Service (`app/services/gemini.py`)
- Query expansion for improved retrieval
- Response generation with context-aware prompting
- Error handling for API failures

#### Retriever (`app/services/retriever.py`)
- Implements the multi-stage retrieval pipeline:
  1. Initial semantic search with original query
  2. Query expansion using Gemini API
  3. Hybrid retrieval combining results
  4. Context optimization and ordering

#### Ranker (`app/services/ranker.py`)
- Custom FlashRank implementation with multiple features:
  - Semantic similarity between query and passage
  - Term overlap and keyword matching
  - Positional bias from initial ranking
  - Query term density in passages

### 3. Core Configuration (`app/core/config.py`)

Manages all configuration through environment variables:
- API settings (title, description, version)
- Gemini API key and model selection
- ChromaDB persistence directory
- Embedding model name
- Retrieval parameters (chunk size, overlap, result counts)

## Multi-Stage Retrieval Pipeline

### Stage 1: Initial Semantic Search
- Converts query to vector using BAAI/bge-small-en-v1.5
- Performs similarity search against indexed documents
- Retrieves initial candidate passages

### Stage 2: Query Expansion
- Uses Gemini API to rewrite and expand the original query
- Captures additional aspects, synonyms, or related concepts
- Improves recall for documents that match intent but not exact wording

### Stage 3: Hybrid Retrieval & Combination
- Performs second search using expanded query
- Combines and deduplicates results from both searches
- Applies differential weighting and sorting by relevance

### Stage 4: FlashRank Re-ranking
- Applies feature-based re-ranking to further improve relevance
- Utilizes multiple ranking signals
- Re-scores passages based on comprehensive relevance assessment

## Context Optimization

Before sending to the LLM, retrieved passages undergo several optimization steps:
- **Relevance Scoring** - Passages scored by similarity and query term matching
- **Deduplication** - Removal of redundant information
- **Citation Management** - Each passage gets a citation number
- **Context Ordering** - Most relevant content prioritized in the context window

## API Endpoints

### Document Upload
```
POST /api/documents
```
Add text documents to the RAG system.

**Request Body:**
```json
{
  "documents": [
    {
      "content": "Text content to be processed..."
    }
  ]
}
```

### PDF Upload
```
POST /api/upload-pdf
```
Upload a PDF document for processing.

**Request:** Form data with PDF file

### Query
```
POST /api/query
```
Query the RAG system with multi-stage retrieval.

**Request Body:**
```json
{
  "query": "Your question here",
  "max_results": 5
}
```

**Response:**
```json
{
  "original_query": "Your question here",
  "expanded_query": "Your expanded question with additional terms",
  "results": [...],
  "answer": "Generated answer based on retrieved context"
}
```

## Setup and Installation

1. Clone the repository
2. Set up a Python environment
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   CHROMA_PERSIST_DIRECTORY=./data/chroma
   MODEL_NAME=BAAI/bge-small-en-v1.5
   ```
5. Run the server:
   ```
   python main.py
   ```
6. Access the API documentation at `http://localhost:8000/docs`

## Usage Example

Check `example_usage.py` for a sample implementation showing how to:

1. Add text documents
2. Upload PDF documents
3. Query the system with multi-stage retrieval

## Future Enhancements

- Enhanced FlashRank re-ranking with more advanced features
- Improved PDF processing with table and image extraction
- Additional output formats and citation styles
- Integration with more LLM providers