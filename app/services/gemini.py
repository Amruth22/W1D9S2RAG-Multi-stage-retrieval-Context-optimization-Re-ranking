import google.generativeai as genai
from app.core.config import settings

class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    async def generate_response(self, prompt, context):
        """Generate a response using Gemini API with the provided context."""
        try:
            full_prompt = f"""Context information:
            {context}
            
            Answer the question based on the context above: {prompt}
            
            If the context doesn't contain relevant information, say "I don't have enough information to answer that."
            """
            
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def rewrite_query(self, original_query):
        """Rewrite a query to improve retrieval (for multi-stage retrieval)."""
        try:
            prompt = f"""Please rewrite the following question to make it more detailed and specific to improve search results.
            Original question: {original_query}
            Rewritten question:"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return original_query