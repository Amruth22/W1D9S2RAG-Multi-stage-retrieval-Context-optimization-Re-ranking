from fastapi import APIRouter, HTTPException, Depends, status, Body, UploadFile, File, Form
from app.api.models import DocumentsIn, QueryIn, QueryResult, StatusResponse, DocumentOut
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import ChromaService
from app.services.retriever import MultiStageRetriever
from app.services.gemini import GeminiService
from app.services.pdf_processor import PDFProcessor
from typing import List
import os

router = APIRouter()

@router.post("/documents", response_model=StatusResponse, tags=["Documents"])
async def add_documents(documents_in: DocumentsIn):
    """
    Add documents to the RAG system.
    
    The documents will be processed, chunked, embedded, and stored in the vector database.
    """
    try:
        processor = DocumentProcessor()
        chroma_service = ChromaService()
        
        total_chunks = 0
        
        for doc in documents_in.documents:
            # Process document into chunks with default metadata
            source_metadata = {"source": "api_upload"}
            chunks, metadatas = await processor.process_document(doc.content, source_metadata)
            
            # Add chunks to vector store
            ids = await chroma_service.add_documents(chunks, metadatas)
            total_chunks += len(chunks)
        
        return {
            "status": "success",
            "message": f"Added {len(documents_in.documents)} documents with {total_chunks} total chunks."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents: {str(e)}"
        )

@router.post("/upload-pdf", response_model=StatusResponse, tags=["Documents"])
async def upload_pdf(pdf_file: UploadFile = File(...)):
    """
    Upload a PDF document to the RAG system.
    
    The PDF will be processed, text extracted, chunked, embedded, and stored in the vector database.
    """
    try:
        # Read the uploaded PDF content
        file_content = await pdf_file.read()
        
        # Process the PDF
        pdf_processor = PDFProcessor()
        chroma_service = ChromaService()
        
        # Extract text and process into chunks
        chunks, metadatas = await pdf_processor.process_pdf(file_content, filename=pdf_file.filename)
        
        # Add chunks to vector store
        ids = await chroma_service.add_documents(chunks, metadatas)
        
        return {
            "status": "success",
            "message": f"PDF '{pdf_file.filename}' processed successfully with {len(chunks)} chunks extracted."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )

@router.post("/query", response_model=QueryResult, tags=["Query"])
async def query(query_in: QueryIn):
    """
    Process a query using the multi-stage RAG system.
    
    This endpoint performs multi-stage retrieval, context optimization, and generates a response.
    """
    try:
        retriever = MultiStageRetriever()
        gemini_service = GeminiService()
        
        # Get optimized context through multi-stage retrieval
        retrieval_result = await retriever.retrieve(query_in.query, query_in.max_results)
        
        # Format context for the LLM
        context = "\n\n".join([
            f"[{result['citation_number']}] {result['content']}"
            for result in retrieval_result["results"]
        ])
        
        # Generate answer with Gemini
        answer = await gemini_service.generate_response(query_in.query, context)
        
        # Prepare response
        return {
            "original_query": retrieval_result["original_query"],
            "expanded_query": retrieval_result["expanded_query"],
            "results": retrieval_result["results"],
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )