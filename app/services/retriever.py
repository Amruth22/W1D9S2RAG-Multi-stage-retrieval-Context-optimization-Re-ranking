from app.services.vector_store import ChromaService
from app.services.gemini import GeminiService
from app.services.ranker import FlashRank
import re

class MultiStageRetriever:
    def __init__(self):
        self.chroma_service = ChromaService()
        self.gemini_service = GeminiService()
        self.ranker = FlashRank()
    
    async def retrieve(self, query, max_results=5):
        """Multi-stage retrieval process."""
        # Stage 1: Initial semantic search
        first_stage_results = await self.chroma_service.similarity_search(
            query, 
            n_results=max_results*2  # Get more results initially
        )
        
        # Stage 2: Query expansion with Gemini
        expanded_query = await self.gemini_service.rewrite_query(query)
        
        # Combine original and expanded queries for better search
        expanded_results = await self.chroma_service.similarity_search(
            expanded_query,
            n_results=max_results*2
        )
        
        # Stage 3: Hybrid retrieval - combine and deduplicate results
        combined_results = self._combine_search_results(
            first_stage_results,
            expanded_results,
            max_results
        )
        
        # Context optimization - simple version (ordering by relevance)
        optimized_context = self._optimize_context(combined_results, query)
        
        # Stage 4: Re-ranking with FlashRank
        reranked_results = await self.ranker.rerank(query, optimized_context, max_results)
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "results": reranked_results
        }
    
    def _combine_search_results(self, first_results, second_results, max_results):
        """Combine and deduplicate results from multiple searches."""
        # Track documents by ID to avoid duplicates
        seen_ids = set()
        combined = []
        
        # Process first results
        for i, doc_id in enumerate(first_results["ids"]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append({
                    "id": doc_id,
                    "content": first_results["documents"][i],
                    "metadata": first_results["metadatas"][i],
                    "score": 1.0 - first_results["distances"][i],  # Convert distance to score
                    "source": "original_query"
                })
        
        # Process second results
        for i, doc_id in enumerate(second_results["ids"]):
            if doc_id not in seen_ids and len(combined) < max_results * 2:
                seen_ids.add(doc_id)
                combined.append({
                    "id": doc_id,
                    "content": second_results["documents"][i],
                    "metadata": second_results["metadatas"][i],
                    "score": 0.9 * (1.0 - second_results["distances"][i]),  # Slightly lower weight
                    "source": "expanded_query"
                })
        
        # Sort by score (descending)
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top results
        return combined[:max_results]
    
    def _optimize_context(self, results, query):
        """Basic context optimization."""
        # Simple optimization: order by relevance and add citation information
        for i, result in enumerate(results):
            # Add citation info
            result["citation_number"] = i + 1
            
            # Extract any query terms to highlight (basic)
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            content_lower = result["content"].lower()
            
            # Relevance signal - count query term occurrences
            term_count = sum(1 for term in query_terms if term in content_lower)
            result["relevance_signals"] = {"query_term_count": term_count}
        
        return results