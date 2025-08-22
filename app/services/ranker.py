from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer, util
from app.core.config import settings
from app.services.embedding import EmbeddingService
import torch
import re

class FlashRank:
    """
    Custom implementation of a re-ranking system for RAG, inspired by cross-encoder re-ranking approaches.
    This class implements features similar to what you might expect from a "Flash Rank" system:
    1. Feature-based re-ranking (lexical and semantic features)
    2. Optimized for fast execution
    3. Context-aware scoring
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        # Weight for different re-ranking features
        self.weights = {
            "semantic_similarity": 0.65,   # Semantic similarity between query and passage
            "term_overlap": 0.15,          # Direct term matches
            "positional_bias": 0.10,       # Position in original results
            "query_term_density": 0.10,    # Density of query terms in passage
        }
    
    async def rerank(self, query: str, initial_results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank passages based on multiple features for more relevant results.
        
        Args:
            query: The user query
            initial_results: List of initial retrieval results with content, metadata, score
            top_k: Number of results to return after re-ranking
            
        Returns:
            List of re-ranked results
        """
        if not initial_results:
            return []
        
        if top_k is None:
            top_k = len(initial_results)
        
        reranked_results = []
        
        # Get query embedding for semantic comparisons
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Preprocess query for term matching
        query_terms = set(self._preprocess_text(query))
        
        # Calculate features for each result
        for i, result in enumerate(initial_results):
            # Initialize scores dictionary
            scores = {}
            
            # 1. Semantic similarity using embeddings
            content_embedding = self.embedding_service.get_embedding(result["content"])
            
            # Calculate cosine similarity
            semantic_similarity = util.cos_sim(
                torch.tensor(query_embedding).unsqueeze(0), 
                torch.tensor(content_embedding).unsqueeze(0)
            ).item()
            scores["semantic_similarity"] = float(semantic_similarity)
            
            # 2. Term overlap - simple term matching
            content_terms = set(self._preprocess_text(result["content"]))
            term_overlap = len(query_terms.intersection(content_terms)) / max(1, len(query_terms))
            scores["term_overlap"] = term_overlap
            
            # 3. Positional bias - reward passages that were ranked higher initially
            position_score = 1.0 / (i + 1)  # Higher score for earlier positions
            scores["positional_bias"] = position_score
            
            # 4. Query term density
            term_density = self._calculate_query_term_density(query_terms, result["content"])
            scores["query_term_density"] = term_density
            
            # Calculate weighted final score
            final_score = sum(scores[key] * self.weights[key] for key in scores)
            
            # Store the scores for explanation
            result["ranking_features"] = scores
            result["rerank_score"] = final_score
            reranked_results.append(result)
        
        # Sort by final re-rank score (descending)
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top_k results
        return reranked_results[:top_k]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for term matching."""
        # Convert to lowercase and split on non-alphanumeric characters
        return [term.lower() for term in re.findall(r'\b\w+\b', text.lower()) if len(term) > 2]
    
    def _calculate_query_term_density(self, query_terms: set, text: str) -> float:
        """Calculate the density of query terms in the text."""
        if not query_terms:
            return 0.0
        
        text_terms = self._preprocess_text(text)
        if not text_terms:
            return 0.0
        
        # Count occurrences of query terms in text
        matches = sum(1 for term in text_terms if term in query_terms)
        
        # Density = matches / text_length
        return matches / len(text_terms)