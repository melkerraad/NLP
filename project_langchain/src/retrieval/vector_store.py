"""Vector store factory for creating and managing LangChain vector stores."""

from pathlib import Path
from typing import Optional, List
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStoreFactory:
    """Factory for creating vector stores with different configurations."""
    
    @staticmethod
    def create_embeddings(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Create embedding function.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            HuggingFaceEmbeddings instance
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fallback to CPU if CUDA requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available for embeddings, falling back to CPU")
            device = "cpu"
        
        print(f"[INFO] Creating embeddings on device: {device}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
    
    @staticmethod
    def create_vector_store(
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        documents: Optional[List[Document]] = None,
        device: Optional[str] = None
    ) -> Chroma:
        """Create or load a ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Name of embedding model
            documents: Optional documents to add (for new stores)
            device: Device to use for embeddings ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            Chroma vector store instance
        """
        # Create embeddings
        embeddings = VectorStoreFactory.create_embeddings(embedding_model, device=device)
        
        # Create persist directory if needed
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
        
        # Check if vector store already exists
        if persist_directory and Path(persist_directory).exists():
            try:
                # Try to load existing vector store
                vector_store = Chroma(
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    embedding_function=embeddings
                )
                
                # Check if collection has documents
                if vector_store._collection.count() > 0:
                    print(f"[OK] Loaded existing vector store with {vector_store._collection.count()} documents")
                    return vector_store
            except Exception as e:
                print(f"[INFO] Could not load existing vector store: {e}")
                print("   Creating new vector store...")
        
        # Create new vector store
        if documents:
            vector_store = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"[OK] Created new vector store with {len(documents)} documents")
        else:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print(f"[OK] Created empty vector store")
        
        return vector_store
    
    @staticmethod
    def load_existing(
        collection_name: str,
        persist_directory: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ) -> Chroma:
        """Load an existing vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory where database is persisted
            embedding_model: Name of embedding model (must match original)
            device: Device to use for embeddings ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            Chroma vector store instance
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        if not Path(persist_directory).exists():
            raise FileNotFoundError(f"Vector store not found at: {persist_directory}")
        
        embeddings = VectorStoreFactory.create_embeddings(embedding_model, device=device)
        
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        count = vector_store._collection.count()
        print(f"[OK] Loaded vector store with {count} documents")
        
        return vector_store

