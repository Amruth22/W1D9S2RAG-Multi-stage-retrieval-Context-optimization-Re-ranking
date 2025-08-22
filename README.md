# Advanced RAG System with Multi-Stage Retrieval

A sophisticated Retrieval-Augmented Generation (RAG) system implementing multi-stage retrieval, context optimization, and preparation for re-ranking with Google's Gemini API integration.

## Project Overview

This project implements an advanced RAG system with the following key components:

1. **Multi-stage retrieval** - A three-stage retrieval pipeline that improves result quality
2. **Context optimization** - Smart techniques to organize and prioritize retrieved passages
3. **PDF processing** - Built-in capabilities to extract and process text from PDF documents
4. **FastAPI interface** - Clean API endpoints with Swagger documentation

## Tech Stack

- **Python 3.9+** - Core programming language
- **FastAPI** - API framework with automatic Swagger documentation
- **BAAI/bge-small-en-v1.5** - Embedding model for text vectorization
- **ChromaDB** - Vector database for similarity search
- **Google Generative AI** - Gemini API for query expansion and response generation
- **PyPDF2 & PDFPlumber** - PDF text extraction libraries
- **LangChain** - Text chunking and document processing tools
- **Uvicorn** - ASGI web server for FastAPI

## Architecture

### Multi-Stage Retrieval

The system implements a sophisticated four-stage retrieval pipeline:

1. **Initial Semantic Search**
   - Converts query to vector using BAAI/bge-small-en-v1.5
   - Performs similarity search against indexed documents
   - Retrieves initial candidate passages

2. **Query Expansion**
   - Uses Gemini API to rewrite and expand the original query
   - Captures additional aspects, synonyms, or related concepts
   - Improves recall for documents that match the intent but not exact wording

3. **Hybrid Retrieval & Combination**
   - Performs second search using expanded query
   - Combines and deduplicates results from both searches
   - Applies differential weighting and sorting by relevance

4. **FlashRank Re-ranking**
   - Applies feature-based re-ranking to further improve relevance
   - Utilizes multiple ranking signals: semantic similarity, term overlap, position bias
   - Re-scores passages based on comprehensive relevance assessment

### Context Optimization & Re-ranking

Retrieved passages undergo optimization and re-ranking before being sent to the LLM:

- **Relevance Scoring** - Passages scored by similarity and query term matching
- **Deduplication** - Removal of redundant information
- **Citation Management** - Each passage gets a citation number
- **Context Ordering** - Most relevant content prioritized in the context window
- **FlashRank Re-ranking** - Multi-feature re-ranking that considers:
  - Semantic similarity between query and passage
  - Term overlap and keyword matching
  - Positional bias from initial ranking
  - Query term density in passages

## API Endpoints

### 1. Document Upload

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

### 2. PDF Upload

```
POST /api/upload-pdf
```
Upload a PDF document for processing.

**Request:** Form data with PDF file

### 3. Query

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