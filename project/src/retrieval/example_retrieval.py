"""Example retrieval system implementation.

This is a template showing how to set up a basic retrieval system.
"""

from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer


class CourseRetriever:
    """Retrieval system for course documents."""
    
    def __init__(self, collection_name: str = "courses"):
        """Initialize the retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
        
        # Initialize embedding model
        # Using a lightweight model for example - you may want to use a better one
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, documents: List[str], metadata: List[Dict], ids: List[str]):
        """Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadata: List of metadata dictionaries
            ids: List of unique document IDs
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to collection")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return retrieved_docs


def main():
    """Example usage."""
    # Initialize retriever
    retriever = CourseRetriever()
    
    # Example: Add some documents
    # In practice, you'll load these from your processed course data
    example_docs = [
        "DAT450: Natural Language Processing. Introduction to NLP...",
        "DAT250: Data Structures and Algorithms. Core CS concepts...",
    ]
    example_metadata = [
        {"course_code": "DAT450", "course_name": "NLP"},
        {"course_code": "DAT250", "course_name": "Data Structures"},
    ]
    example_ids = ["doc1", "doc2"]
    
    retriever.add_documents(example_docs, example_metadata, example_ids)
    
    # Example: Retrieve documents
    query = "What courses cover machine learning?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} documents:")
    for result in results:
        print(f"- {result['document'][:100]}...")


if __name__ == "__main__":
    main()

