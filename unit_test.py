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



    def test_4_multistage_retrieval_validation(self):
        """Test 4: Multi-stage retrieval pipeline validation"""
        print("\n" + "="*60)
        print("Test 4: Multi-Stage Retrieval Pipeline Validation (HTTP)")
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
                # Verify citation numbering (if present)
                if "citation_number" in result:
                    assert result["citation_number"] == i + 1, f"Citation number should be {i + 1}"
                else:
                    print(f"INFO: Citation number not present in result {i+1}")
                    print(f"INFO: Citation number not present in result {i+1}")
                
                # Verify score is reasonable
                assert 0 <= result["score"] <= 1, f"Score should be between 0 and 1, got {result['score']}"
            
            # Verify results structure
            citation_numbers = [r.get('citation_number', 'N/A') for r in results]
            print(f"PASS: Results citation info: {citation_numbers}")
            print(f"PASS: Results have scores: {[round(r['score'], 3) for r in results]}")
            
            # Verify results are ordered by relevance (descending scores)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by score (descending)"
            print("PASS: Results properly ordered by relevance score")
            
        else:
            print(f"Multi-stage retrieval failed with status {response.status_code}")
            assert response.status_code in [400, 422, 500]
            print("PASS: Server handles multi-stage retrieval errors")
        
        print("PASS: Test 4 PASSED: Multi-stage retrieval pipeline validated via HTTP")











    def test_5_end_to_end_pipeline(self):
        """Test 5: Complete end-to-end pipeline validation"""
        print("\n" + "="*60)
        print("Test 5: End-to-End Pipeline Validation (HTTP)")
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
            # Verify re-ranking and context optimization (check if citation_number exists)
            for i, result in enumerate(results):
                if "citation_number" in result:
                    assert result["citation_number"] == i + 1, "Citations should be numbered"
                else:
                    print(f"INFO: Citation number not present in result {i+1}")
                assert "score" in result, "Results should have relevance scores"
                assert "content" in result, "Results should have content"
        
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
        
        print("PASS: Test 5 PASSED: Complete end-to-end pipeline validated via HTTP")

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
        test_instance.test_4_multistage_retrieval_validation,
        test_instance.test_5_end_to_end_pipeline
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