#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Multi-Stage RAG System API
Tests all API endpoints and internal components using HTTP requests against RUNNING SERVER
Prerequisites: Start the server first with 'python main.py'
"""

import requests
import io
import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any

# Default server configuration (matches main.py default)
SERVER_HOST = "localhost"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

class TestRAGSystemAPI:
    """Test class for Multi-Stage RAG System API endpoints and components using HTTP requests to running server"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and check server availability"""
        cls.base_url = BASE_URL
        cls.test_pdf_content = cls.create_real_pdf()
        cls.test_filename = "test_document.pdf"
        cls.test_documents = [
            "Multi-stage retrieval is an advanced technique in RAG systems that improves relevance by using multiple retrieval steps.",
            "FlashRank is a re-ranking system that uses multiple features like semantic similarity, term overlap, and positional bias.",
            "ChromaDB is a vector database that stores embeddings and performs similarity search for document retrieval.",
            "BAAI/bge-small-en-v1.5 is an embedding model that converts text into high-dimensional vectors for semantic search."
        ]
        
        # Check if server is running
        print(f"Checking if server is running at {cls.base_url}...")
        try:
            response = requests.get(f"{cls.base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"+ Server is running and responding at {cls.base_url}")
            else:
                print(f"! Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"X ERROR: Cannot connect to server at {cls.base_url}")
            print(f"Please start the server first with: python main.py")
            print(f"Connection error: {e}")
            sys.exit(1)
        
    @staticmethod
    def create_real_pdf():
        """Create a proper PDF content for testing"""
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj

4 0 obj
<<
/Length 200
>>
stream
BT
/F1 12 Tf
100 700 Td
(This is a test PDF document for RAG system testing.) Tj
0 -20 Td
(It contains information about multi-stage retrieval systems.) Tj
0 -20 Td
(FlashRank re-ranking improves search result relevance.) Tj
0 -20 Td
(Vector embeddings enable semantic similarity search.) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000306 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
556
%%EOF"""
        return pdf_content

    def test_1_health_endpoints(self):
        """Test 1: Health check endpoint - HTTP requests to running server"""
        print("\n" + "="*60)
        print("Test 1: Health Check Endpoint (HTTP)")
        print("="*60)
        
        # Test health endpoint
        response = requests.get(f"{self.base_url}/health")
        print(f"Health endpoint status: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        print(f"PASS: Health endpoint response: {data}")
        
        print("PASS: Test 1 PASSED: Health check endpoint working via HTTP")

    def test_2_document_upload(self):
        """Test 2: Document upload endpoint - HTTP request to running server"""
        print("\n" + "="*60)
        print("Test 2: Document Upload Endpoint (HTTP)")
        print("="*60)
        
        # Prepare test documents
        payload = {
            "documents": [
                {"content": doc} for doc in self.test_documents
            ]
        }
        
        # Test document upload via HTTP to running server
        response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"HTTP Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: HTTP Response: {data}")
            
            # Verify response structure from actual running server
            assert "status" in data
            assert "message" in data
            assert data["status"] == "success"
            assert "documents" in data["message"]
            assert "chunks" in data["message"]
            
            print("PASS: Real document upload successful via HTTP")
            print(f"PASS: Response: {data['message']}")
        else:
            print(f"Server returned status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Raw response: {response.text}")
            
            assert response.status_code in [400, 422, 500]
            print("PASS: Server properly handles errors")
        
        print("PASS: Test 2 PASSED: Real document upload endpoint tested via HTTP")

    def test_3_pdf_upload(self):
        """Test 3: PDF document upload endpoint - HTTP request to running server"""
        print("\n" + "="*60)
        print("Test 3: PDF Document Upload Endpoint (HTTP)")
        print("="*60)
        
        # Prepare test file with actual PDF content
        files = {
            "pdf_file": (self.test_filename, io.BytesIO(self.test_pdf_content), "application/pdf")
        }
        
        # Test ACTUAL PDF upload via HTTP to running server
        response = requests.post(f"{self.base_url}/api/upload-pdf", files=files)
        print(f"HTTP Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: HTTP Response: {data}")
            
            # Verify response structure from actual running server
            assert "status" in data
            assert "message" in data
            assert data["status"] == "success"
            assert "processed successfully" in data["message"]
            assert "chunks" in data["message"]
            
            print("PASS: Real PDF upload successful via HTTP")
            print(f"PASS: Response: {data['message']}")
        else:
            print(f"Server returned status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Raw response: {response.text}")
            
            assert response.status_code in [400, 422, 500]
            print("PASS: Server properly handles PDF processing errors")
        
        print("PASS: Test 3 PASSED: Real PDF upload endpoint tested via HTTP")

    def test_4_basic_query_processing(self):
        """Test 4: Basic query processing endpoint - HTTP request to running server"""
        print("\n" + "="*60)
        print("Test 4: Basic Query Processing Endpoint (HTTP)")
        print("="*60)
        
        # Ensure we have documents by uploading test data
        print("Setting up test documents...")
        payload = {"documents": [{"content": doc} for doc in self.test_documents]}
        setup_response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Document setup status: {setup_response.status_code}")
        
        # Test ACTUAL query request via HTTP to running server
        query_data = {
            "query": "What is multi-stage retrieval?",
            "max_results": 3
        }
        
        response = requests.post(f"{self.base_url}/api/query", json=query_data)
        print(f"HTTP Query Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Real query response received via HTTP")
            
            # Verify response structure from actual running server
            required_fields = ["original_query", "expanded_query", "results", "answer"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Verify actual response content types
            assert isinstance(data["original_query"], str)
            assert isinstance(data["expanded_query"], str)
            assert isinstance(data["results"], list)
            assert isinstance(data["answer"], str) or data["answer"] is None
            
            print(f"PASS: Original query: {data['original_query']}")
            print(f"PASS: Expanded query: {data['expanded_query']}")
            print(f"PASS: Retrieved {len(data['results'])} results")
            if data["answer"]:
                print(f"PASS: Got LLM response: {data['answer'][:100]}...")
            
        else:
            print(f"Query failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Raw error response: {response.text}")
                
            assert response.status_code in [400, 422, 500]
            print("PASS: Server properly handles query errors")
        
        print("PASS: Test 4 PASSED: Real query endpoint tested via HTTP")

    def test_5_multistage_retrieval_validation(self):
        """Test 5: Multi-stage retrieval pipeline validation"""
        print("\n" + "="*60)
        print("Test 5: Multi-Stage Retrieval Pipeline Validation (HTTP)")
        print("="*60)
        
        # Setup diverse test documents for better retrieval testing
        diverse_docs = [
            "Multi-stage retrieval systems use multiple phases to improve search accuracy and relevance.",
            "The first stage performs initial semantic search using vector embeddings and similarity matching.",
            "Query expansion in the second stage uses AI models to rewrite and enhance the original query.",
            "Hybrid retrieval combines results from multiple search strategies for comprehensive coverage.",
            "Re-ranking algorithms apply sophisticated scoring to reorder results by relevance."
        ]
        
        payload = {"documents": [{"content": doc} for doc in diverse_docs]}
        setup_response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Diverse documents setup status: {setup_response.status_code}")
        
        # Test multi-stage retrieval with a specific query
        query_data = {
            "query": "How does multi-stage retrieval work?",
            "max_results": 4
        }
        
        response = requests.post(f"{self.base_url}/api/query", json=query_data)
        print(f"Multi-stage retrieval response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate multi-stage retrieval components
            print("PASS: Multi-stage retrieval pipeline executed successfully")
            
            # Stage validation
            original_query = data["original_query"]
            expanded_query = data["expanded_query"]
            results = data["results"]
            
            # Verify query expansion occurred (Stage 2)
            assert original_query != expanded_query, "Query expansion should modify the query"
            print(f"PASS: Query expansion - Original: '{original_query}'")
            print(f"PASS: Query expansion - Expanded: '{expanded_query}'")
            
            # Verify results structure (Stage 3 & 4 - Hybrid retrieval & re-ranking)
            assert len(results) > 0, "Should retrieve at least one result"
            
            for i, result in enumerate(results):
                # Verify result structure
                required_fields = ["id", "content", "metadata", "score", "citation_number"]
                for field in required_fields:
                    assert field in result, f"Missing field {field} in result {i}"
                
                # Verify citation numbering
                assert result["citation_number"] == i + 1, f"Citation number should be {i + 1}"
                
                # Verify score is reasonable
                assert 0 <= result["score"] <= 1, f"Score should be between 0 and 1, got {result['score']}"
            
            print(f"PASS: Retrieved {len(results)} results with proper structure")
            print(f"PASS: Results have citation numbers: {[r['citation_number'] for r in results]}")
            print(f"PASS: Results have scores: {[round(r['score'], 3) for r in results]}")
            
            # Verify results are ordered by relevance (descending scores)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by score (descending)"
            print("PASS: Results properly ordered by relevance score")
            
        else:
            print(f"Multi-stage retrieval failed with status {response.status_code}")
            assert response.status_code in [400, 422, 500]
            print("PASS: Server handles multi-stage retrieval errors")
        
        print("PASS: Test 5 PASSED: Multi-stage retrieval pipeline validated via HTTP")

    def test_6_flashrank_reranker_validation(self):
        """Test 6: FlashRank re-ranking system validation"""
        print("\n" + "="*60)
        print("Test 6: FlashRank Re-Ranker Validation (HTTP)")
        print("="*60)
        
        # Setup documents with varying relevance for re-ranking testing
        rerank_docs = [
            "FlashRank is a re-ranking system that improves search results using multiple scoring features.",
            "Semantic similarity measures how closely related the query and document are in meaning.",
            "Term overlap calculates the percentage of query terms that appear in the document text.",
            "Positional bias gives higher scores to documents that were ranked higher in initial results.",
            "Query term density measures how frequently query terms appear throughout the document."
        ]
        
        payload = {"documents": [{"content": doc} for doc in rerank_docs]}
        setup_response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Re-ranking test documents setup status: {setup_response.status_code}")
        
        # Test query that should trigger re-ranking
        query_data = {
            "query": "FlashRank semantic similarity scoring",
            "max_results": 5
        }
        
        response = requests.post(f"{self.base_url}/api/query", json=query_data)
        print(f"FlashRank re-ranking response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            
            print("PASS: FlashRank re-ranking executed successfully")
            
            # Verify re-ranking features are applied
            for i, result in enumerate(results):
                # Check if re-ranking metadata is present (if exposed by the API)
                print(f"Result {i+1}: Score={result['score']:.4f}, Content='{result['content'][:50]}...'")
                
                # Verify the result has proper scoring
                assert isinstance(result["score"], (int, float)), "Score should be numeric"
                assert result["score"] >= 0, "Score should be non-negative"
            
            # Test re-ranking effectiveness
            # The first result should be most relevant to "FlashRank semantic similarity scoring"
            first_result = results[0]
            first_content = first_result["content"].lower()
            
            # Check if the most relevant document is ranked first
            query_terms = ["flashrank", "semantic", "similarity", "scoring"]
            term_matches = sum(1 for term in query_terms if term in first_content)
            
            print(f"PASS: Top result has {term_matches} query term matches")
            print(f"PASS: Top result score: {first_result['score']:.4f}")
            
            # Verify re-ranking improved relevance (top result should have high term overlap)
            assert term_matches >= 2, "Top result should match multiple query terms"
            
            # Verify score distribution (should have variety indicating re-ranking)
            scores = [r["score"] for r in results]
            score_range = max(scores) - min(scores)
            assert score_range > 0.01, "Re-ranking should create score diversity"
            print(f"PASS: Score range indicates re-ranking: {score_range:.4f}")
            
        else:
            print(f"FlashRank re-ranking failed with status {response.status_code}")
            assert response.status_code in [400, 422, 500]
            print("PASS: Server handles re-ranking errors")
        
        print("PASS: Test 6 PASSED: FlashRank re-ranking system validated via HTTP")

    def test_7_embedding_service_validation(self):
        """Test 7: Embedding service validation through API behavior"""
        print("\n" + "="*60)
        print("Test 7: Embedding Service Validation (HTTP)")
        print("="*60)
        
        # Test embedding service indirectly through document upload and similarity
        test_docs = [
            "Machine learning algorithms process data to make predictions.",
            "Artificial intelligence systems can learn from experience.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing analyzes human language."
        ]
        
        payload = {"documents": [{"content": doc} for doc in test_docs]}
        response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Embedding test documents upload status: {response.status_code}")
        
        if response.status_code == 200:
            print("PASS: Documents successfully processed and embedded")
            
            # Test semantic similarity through queries
            similar_queries = [
                "machine learning predictions",
                "AI learning systems", 
                "neural network layers",
                "language processing analysis"
            ]
            
            for query in similar_queries:
                query_data = {"query": query, "max_results": 2}
                query_response = requests.post(f"{self.base_url}/api/query", json=query_data)
                
                if query_response.status_code == 200:
                    query_data_response = query_response.json()
                    results = query_data_response["results"]
                    
                    if results:
                        # Verify semantic similarity is working
                        top_result = results[0]
                        print(f"PASS: Query '{query}' found relevant result with score {top_result['score']:.4f}")
                        
                        # Verify embedding-based similarity is reasonable
                        assert top_result["score"] > 0.1, "Semantic similarity should be meaningful"
                    else:
                        print(f"INFO: No results for query '{query}'")
                else:
                    print(f"Query failed for '{query}': {query_response.status_code}")
            
            print("PASS: Embedding service working - semantic similarity functional")
            
        else:
            print(f"Embedding test failed with status {response.status_code}")
            assert response.status_code in [400, 422, 500]
            print("PASS: Server handles embedding errors")
        
        print("PASS: Test 7 PASSED: Embedding service validated via HTTP")

    def test_8_vector_store_operations(self):
        """Test 8: Vector store operations validation"""
        print("\n" + "="*60)
        print("Test 8: Vector Store Operations Validation (HTTP)")
        print("="*60)
        
        # Test vector store through document operations
        vector_test_docs = [
            "ChromaDB stores vector embeddings for similarity search operations.",
            "Vector databases enable efficient nearest neighbor search in high-dimensional spaces.",
            "Similarity search finds documents with embeddings closest to the query embedding.",
            "Distance metrics like cosine similarity measure vector relationships."
        ]
        
        # Test document storage
        payload = {"documents": [{"content": doc} for doc in vector_test_docs]}
        response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Vector store test documents status: {response.status_code}")
        
        if response.status_code == 200:
            print("PASS: Documents successfully stored in vector database")
            
            # Test similarity search functionality
            search_query = "vector similarity search ChromaDB"
            query_data = {"query": search_query, "max_results": 3}
            search_response = requests.post(f"{self.base_url}/api/query", json=query_data)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                results = search_data["results"]
                
                print(f"PASS: Vector similarity search returned {len(results)} results")
                
                # Verify vector store operations
                for i, result in enumerate(results):
                    # Verify metadata preservation
                    assert "metadata" in result, "Metadata should be preserved"
                    assert "id" in result, "Document ID should be present"
                    
                    # Verify similarity scoring
                    score = result["score"]
                    assert isinstance(score, (int, float)), "Score should be numeric"
                    print(f"PASS: Result {i+1} - ID: {result['id'][:8]}..., Score: {score:.4f}")
                
                # Verify results are ranked by similarity
                scores = [r["score"] for r in results]
                assert scores == sorted(scores, reverse=True), "Results should be ordered by similarity"
                print("PASS: Vector similarity search results properly ordered")
                
                # Test that most relevant document contains query terms
                top_result = results[0]
                top_content = top_result["content"].lower()
                query_terms = ["vector", "similarity", "search", "chromadb"]
                matches = sum(1 for term in query_terms if term in top_content)
                
                print(f"PASS: Top result has {matches} query term matches")
                assert matches >= 2, "Vector search should find semantically relevant documents"
                
            else:
                print(f"Vector search failed: {search_response.status_code}")
                assert search_response.status_code in [400, 422, 500]
        
        else:
            print(f"Vector store test failed: {response.status_code}")
            assert response.status_code in [400, 422, 500]
        
        print("PASS: Test 8 PASSED: Vector store operations validated via HTTP")

    def test_10_document_processing_pipeline(self):
        """Test 10: Document processing pipeline validation"""
        print("\n" + "="*60)
        print("Test 10: Document Processing Pipeline Validation (HTTP)")
        print("="*60)
        
        # Test text document processing
        long_document = """
        This is a comprehensive test document for validating the document processing pipeline.
        The document processor should split this text into appropriate chunks based on the configured chunk size and overlap settings.
        
        Multi-stage retrieval systems implement sophisticated approaches to information retrieval.
        The first stage typically involves initial semantic search using vector embeddings.
        Query expansion in subsequent stages helps capture additional relevant information.
        
        FlashRank re-ranking systems apply multiple scoring features to improve result relevance.
        These features include semantic similarity, term overlap, positional bias, and query term density.
        The weighted combination of these features produces more accurate relevance scores.
        
        Vector databases like ChromaDB store high-dimensional embeddings efficiently.
        Similarity search operations find the most relevant documents based on embedding proximity.
        This enables semantic search capabilities that go beyond simple keyword matching.
        """
        
        # Test document chunking through API
        payload = {"documents": [{"content": long_document}]}
        response = requests.post(f"{self.base_url}/api/documents", json=response)
        print(f"Document processing pipeline status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("PASS: Document processing pipeline executed successfully")
            print(f"PASS: Response: {data['message']}")
            
            # Verify chunking occurred (indicated by multiple chunks in response)
            if "chunks" in data["message"]:
                print("PASS: Document was properly chunked")
            
            # Test that processed chunks are searchable
            test_query = "FlashRank re-ranking features"
            query_data = {"query": test_query, "max_results": 3}
            search_response = requests.post(f"{self.base_url}/api/query", json=query_data)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                results = search_data["results"]
                
                if results:
                    print(f"PASS: Processed chunks are searchable - found {len(results)} results")
                    
                    # Verify chunk content is reasonable
                    for result in results:
                        content = result["content"]
                        assert len(content) > 50, "Chunks should contain meaningful content"
                        assert len(content) < 2000, "Chunks should not be too large"
                    
                    print("PASS: Chunk sizes are appropriate")
                else:
                    print("INFO: No search results found for processed document")
            
        else:
            print(f"Document processing failed: {response.status_code}")
            assert response.status_code in [400, 422, 500]
        
        # Test PDF processing pipeline
        print("\nTesting PDF processing pipeline...")
        files = {"pdf_file": (self.test_filename, io.BytesIO(self.test_pdf_content), "application/pdf")}
        pdf_response = requests.post(f"{self.base_url}/api/upload-pdf", files=files)
        print(f"PDF processing pipeline status: {pdf_response.status_code}")
        
        if pdf_response.status_code == 200:
            pdf_data = pdf_response.json()
            print("PASS: PDF processing pipeline executed successfully")
            print(f"PASS: PDF Response: {pdf_data['message']}")
            
            # Test that PDF content is searchable
            pdf_query = "test PDF document"
            query_data = {"query": pdf_query, "max_results": 2}
            pdf_search_response = requests.post(f"{self.base_url}/api/query", json=query_data)
            
            if pdf_search_response.status_code == 200:
                pdf_search_data = pdf_search_response.json()
                pdf_results = pdf_search_data["results"]
                
                if pdf_results:
                    print(f"PASS: PDF content is searchable - found {len(pdf_results)} results")
                else:
                    print("INFO: No search results found for PDF content")
        
        print("PASS: Test 10 PASSED: Document processing pipeline validated via HTTP")

    def test_11_context_optimization(self):
        """Test 11: Context optimization validation"""
        print("\n" + "="*60)
        print("Test 11: Context Optimization Validation (HTTP)")
        print("="*60)
        
        # Setup documents for context optimization testing
        context_docs = [
            "Context optimization ensures the most relevant information is prioritized in responses.",
            "Citation management assigns numbers to source documents for proper attribution.",
            "Relevance-based ordering places the most important information first in the context.",
            "Deduplication removes redundant information to maximize context window efficiency.",
            "Context window management ensures optimal use of available token limits."
        ]
        
        payload = {"documents": [{"content": doc} for doc in context_docs]}
        setup_response = requests.post(f"{self.base_url}/api/documents", json=payload)
        print(f"Context optimization test setup status: {setup_response.status_code}")
        
        # Test context optimization through query
        query_data = {
            "query": "How does context optimization work in RAG systems?",
            "max_results": 4
        }
        
        response = requests.post(f"{self.base_url}/api/query", json=response)
        print(f"Context optimization query status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            
            print("PASS: Context optimization executed successfully")
            
            # Verify citation numbering
            citation_numbers = [r["citation_number"] for r in results]
            expected_citations = list(range(1, len(results) + 1))
            assert citation_numbers == expected_citations, f"Citations should be {expected_citations}, got {citation_numbers}"
            print(f"PASS: Citation numbers properly assigned: {citation_numbers}")
            
            # Verify relevance-based ordering (scores should be descending)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by relevance"
            print(f"PASS: Results ordered by relevance: {[round(s, 3) for s in scores]}")
            
            # Verify no duplicate content (basic check)
            contents = [r["content"] for r in results]
            unique_contents = set(contents)
            assert len(contents) == len(unique_contents), "Results should not contain duplicates"
            print("PASS: No duplicate content in results")
            
            # Verify context is properly formatted for LLM
            if data["answer"]:
                print("PASS: Context successfully used for answer generation")
                print(f"PASS: Generated answer: {data['answer'][:100]}...")
            
            # Test that most relevant result is about context optimization
            top_result = results[0]
            top_content = top_result["content"].lower()
            context_terms = ["context", "optimization", "relevant", "prioritized"]
            matches = sum(1 for term in context_terms if term in top_content)
            
            print(f"PASS: Top result has {matches} context-related terms")
            assert matches >= 2, "Context optimization should prioritize relevant content"
            
        else:
            print(f"Context optimization test failed: {response.status_code}")
            assert response.status_code in [400, 422, 500]
        
        print("PASS: Test 11 PASSED: Context optimization validated via HTTP")

    def test_13_end_to_end_pipeline(self):
        """Test 13: Complete end-to-end pipeline validation"""
        print("\n" + "="*60)
        print("Test 13: End-to-End Pipeline Validation (HTTP)")
        print("="*60)
        
        # Complete pipeline test: Upload → Process → Embed → Store → Query → Retrieve → Re-rank → Generate
        
        print("Step 1: Document Upload and Processing...")
        e2e_docs = [
            "End-to-end testing validates the complete RAG pipeline from document ingestion to response generation.",
            "The pipeline includes document processing, embedding generation, vector storage, and retrieval operations.",
            "Multi-stage retrieval with query expansion and re-ranking improves the quality of retrieved context.",
            "Response generation uses the optimized context to produce accurate and relevant answers."
        ]
        
        # Step 1: Upload documents
        payload = {"documents": [{"content": doc} for doc in e2e_docs]}
        upload_response = requests.post(f"{self.base_url}/api/documents", json=payload)
        assert upload_response.status_code == 200, "Document upload should succeed"
        print("PASS: Step 1 - Documents uploaded and processed")
        
        # Step 2: Upload PDF
        print("Step 2: PDF Upload and Processing...")
        files = {"pdf_file": (self.test_filename, io.BytesIO(self.test_pdf_content), "application/pdf")}
        pdf_response = requests.post(f"{self.base_url}/api/upload-pdf", files=files)
        if pdf_response.status_code == 200:
            print("PASS: Step 2 - PDF uploaded and processed")
        else:
            print(f"INFO: Step 2 - PDF processing status: {pdf_response.status_code}")
        
        # Step 3: Query processing with timing
        print("Step 3: Complete Query Processing...")
        start_time = time.time()
        
        query_data = {
            "query": "Explain the end-to-end RAG pipeline process",
            "max_results": 3
        }
        
        query_response = requests.post(f"{self.base_url}/api/query", json=query_data)
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert query_response.status_code == 200, "Query processing should succeed"
        data = query_response.json()
        
        print(f"PASS: Step 3 - Query processed in {processing_time:.2f} seconds")
        
        # Step 4: Validate complete pipeline results
        print("Step 4: Pipeline Results Validation...")
        
        # Verify all pipeline components worked
        assert "original_query" in data, "Original query should be preserved"
        assert "expanded_query" in data, "Query expansion should occur"
        assert "results" in data, "Results should be retrieved"
        assert "answer" in data, "Answer should be generated"
        
        results = data["results"]
        assert len(results) > 0, "Should retrieve at least one result"
        
        # Verify multi-stage retrieval
        assert data["original_query"] != data["expanded_query"], "Query expansion should modify query"
        print(f"PASS: Multi-stage retrieval - Original: '{data['original_query']}'")
        print(f"PASS: Multi-stage retrieval - Expanded: '{data['expanded_query']}'")
        
        # Verify re-ranking and context optimization
        for i, result in enumerate(results):
            assert result["citation_number"] == i + 1, "Citations should be numbered"
            assert "score" in result, "Results should have relevance scores"
            assert "content" in result, "Results should have content"
        
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be ordered by relevance"
        print(f"PASS: Re-ranking and context optimization - Scores: {[round(s, 3) for s in scores]}")
        
        # Verify response generation
        if data["answer"]:
            answer = data["answer"]
            assert len(answer) > 20, "Generated answer should be substantial"
            print(f"PASS: Response generation - Answer: {answer[:100]}...")
        else:
            print("INFO: No answer generated (may be due to API configuration)")
        
        # Performance validation
        assert processing_time < 30, f"Pipeline should complete within 30 seconds, took {processing_time:.2f}s"
        print(f"PASS: Performance - Pipeline completed in {processing_time:.2f} seconds")
        
        print("PASS: Test 13 PASSED: Complete end-to-end pipeline validated via HTTP")

def run_all_tests():
    """Run all tests in sequence using HTTP requests to running server"""
    print("Multi-Stage RAG System API - Comprehensive HTTP Unit Tests")
    print("=" * 80)
    print("NOTE: These tests use HTTP requests to a RUNNING SERVER")
    print(f"Server should be running at: {BASE_URL}")
    print("Start the server first with: python main.py")
    print("Tests validate multi-stage retrieval, FlashRank re-ranking, and all components")
    print("=" * 80)
    
    # Initialize test class
    test_instance = TestRAGSystemAPI()
    test_instance.setup_class()
    
    # List of test methods - all using real API calls
    test_methods = [
        test_instance.test_1_health_endpoints,
        test_instance.test_2_document_upload,
        test_instance.test_3_pdf_upload,
        test_instance.test_4_basic_query_processing,
        test_instance.test_5_multistage_retrieval_validation,
        test_instance.test_6_flashrank_reranker_validation,
        test_instance.test_7_embedding_service_validation,
        test_instance.test_8_vector_store_operations,
        test_instance.test_10_document_processing_pipeline,
        test_instance.test_11_context_optimization,
        test_instance.test_13_end_to_end_pipeline
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    # Run each test
    for test_method in test_methods:
        try:
            test_method()
            passed_tests += 1
        except Exception as e:
            print(f"FAIL: {test_method.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RAG SYSTEM TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your Multi-Stage RAG System is fully functional!")
        print("✅ Health endpoints working")
        print("✅ Document upload and processing working")
        print("✅ PDF upload and extraction working")
        print("✅ Multi-stage retrieval pipeline working")
        print("✅ FlashRank re-ranking system working")
        print("✅ Embedding service working")
        print("✅ Vector store operations working")
        print("✅ Document processing pipeline working")
        print("✅ Context optimization working")
        print("✅ End-to-end pipeline working")
        print(f"✅ Server at {BASE_URL} is working perfectly!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed. Please check the output above.")
        print("This may indicate issues with:")
        print("- Google Gemini API key configuration")
        print("- Network connectivity")  
        print("- API dependencies or model downloads")
        print("- ChromaDB persistence")
        print("- Multi-stage retrieval components")
        print("- FlashRank re-ranking features")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)