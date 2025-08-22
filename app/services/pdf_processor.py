import PyPDF2
import pdfplumber
from io import BytesIO
import os
from typing import List, Dict, Any, Tuple
from app.services.document_processor import DocumentProcessor

class PDFProcessor:
    def __init__(self):
        self.document_processor = DocumentProcessor()
    
    async def process_pdf(self, file_content: bytes, filename: str = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a PDF file into text chunks for embedding.
        
        Args:
            file_content: The binary content of the PDF file
            filename: Optional filename for metadata
            
        Returns:
            Tuple of (chunks, metadatas)
        """
        try:
            # First attempt with PyPDF2
            text = self._extract_with_pypdf2(file_content)
            
            # If PyPDF2 extraction is too short, try with pdfplumber
            if len(text.strip()) < 100:
                text = self._extract_with_pdfplumber(file_content)
            
            # Create basic metadata
            metadata = {
                "source": filename if filename else "uploaded_pdf",
                "file_type": "pdf"
            }
            
            # Process document into chunks
            chunks, metadatas = await self.document_processor.process_document(text, metadata)
            
            return chunks, metadatas
        
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    def _extract_with_pypdf2(self, file_content: bytes) -> str:
        """Extract text from PDF using PyPDF2."""
        pdf_file = BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        return text
    
    def _extract_with_pdfplumber(self, file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber (more robust but slower)."""
        pdf_file = BytesIO(file_content)
        text = ""
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or "" + "\n\n"
        
        return text