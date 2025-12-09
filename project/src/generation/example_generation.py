"""Example generation system implementation.

This is a template showing how to integrate LLMs for response generation.
"""

from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGGenerator:
    """Generation system for RAG responses."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the generator.
        
        Args:
            model: LLM model name to use
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt with context and query.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        # Format context documents
        context_text = "\n\n".join([
            f"Course: {doc.get('metadata', {}).get('course_code', 'Unknown')}\n"
            f"Content: {doc['document']}"
            for doc in context_docs
        ])
        
        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions about university courses.

Use the following course information to answer the question. If the information is not available in the context, say so.

Course Information:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using LLM.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Generated response string
        """
        # Build prompt
        prompt = self.build_prompt(query, context_docs)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for university course information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def generate_with_sources(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate response with source citations.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        answer = self.generate(query, context_docs)
        
        # Extract source course codes
        sources = [
            doc.get('metadata', {}).get('course_code', 'Unknown')
            for doc in context_docs
        ]
        
        return {
            'answer': answer,
            'sources': list(set(sources))  # Remove duplicates
        }


def main():
    """Example usage."""
    # Initialize generator
    generator = RAGGenerator()
    
    # Example context documents (normally from retrieval system)
    example_context = [
        {
            'document': 'DAT450 covers natural language processing, including transformers and LLMs.',
            'metadata': {'course_code': 'DAT450'}
        },
        {
            'document': 'DAT250 covers fundamental data structures and algorithms.',
            'metadata': {'course_code': 'DAT250'}
        }
    ]
    
    # Generate response
    query = "What does DAT450 cover?"
    result = generator.generate_with_sources(query, example_context)
    
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY in .env file
    main()

