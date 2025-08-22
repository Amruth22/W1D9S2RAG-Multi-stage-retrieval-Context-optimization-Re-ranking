import requests
import os

# API base URL - change if your server is running on a different address
BASE_URL = "http://localhost:8000/api"

def upload_pdf_document(pdf_path):
    """
    Upload a PDF document to the RAG system
    """
    print(f"Uploading PDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return False
    
    # Prepare the file for upload
    with open(pdf_path, 'rb') as f:
        files = {'pdf_file': (os.path.basename(pdf_path), f, 'application/pdf')}
        
        # Send request to upload endpoint
        response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
    
    # Check response
    if response.status_code == 200:
        print(f"PDF uploaded successfully: {response.json()}")
        return True
    else:
        print(f"Error uploading PDF: {response.status_code} - {response.text}")
        return False

def add_text_document(text):
    """
    Add a text document to the RAG system
    """
    print("Adding text document")
    
    # Prepare request
    payload = {
        "documents": [
            {
                "content": text
            }
        ]
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/documents", json=payload)
    
    # Check response
    if response.status_code == 200:
        print(f"Document added successfully: {response.json()}")
        return True
    else:
        print(f"Error adding document: {response.status_code} - {response.text}")
        return False

def query_rag_system(query, max_results=5):
    """
    Query the RAG system
    """
    print(f"Querying: '{query}'")
    
    # Prepare request
    payload = {
        "query": query,
        "max_results": max_results
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/query", json=payload)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"\nOriginal Query: {result['original_query']}")
        print(f"Expanded Query: {result['expanded_query']}")
        print(f"\nAnswer: {result['answer']}")
        
        print("\nSource Chunks:")
        for i, chunk in enumerate(result['results']):
            print(f"[{i+1}] {chunk['content'][:100]}... (score: {chunk['score']:.4f})")
        
        return result
    else:
        print(f"Error querying system: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    print("RAG System Example Usage\n")
    print("Make sure your FastAPI server is running!\n")
    
    # 1. Add a text document
    sample_text = """
    Multi-stage retrieval is an advanced technique in RAG systems that improves relevance by using 
    multiple retrieval steps. In the first stage, an initial set of candidate documents is retrieved,
    often using fast but less precise methods. In subsequent stages, these candidates are refined,
    reranked, or expanded to improve relevance. This approach combines the efficiency of simple retrieval
    with the precision of more sophisticated methods, ultimately providing better context to the LLM.
    """
    
    add_text_document(sample_text)
    
    # 2. Query the system
    query = "What is multi-stage retrieval?"
    result = query_rag_system(query)
    
    # 3. Instructions for uploading a PDF
    print("\nTo upload a PDF:")
    print("upload_pdf_document('path/to/your/document.pdf')")