"""Test script for LangChain RAG pipeline."""

from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStoreFactory
from src.generation.llm_factory import create_llm_from_config
from src.generation.rag_chain import create_rag_chain


def main():
    """Run RAG pipeline tests."""
    print("=" * 70)
    print("LangChain RAG Pipeline Test")
    print("=" * 70)
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    config = load_config()
    print("[OK] Configuration loaded")
    
    # Load vector store
    print("\n[2/4] Loading vector store...")
    vector_config = config.get_vector_store_config()
    retrieval_config = config.get_retrieval_config()
    project_root = Path(__file__).parent
    
    persist_dir = str(project_root / vector_config.get("persist_directory", "data/chroma_db"))
    
    try:
        vector_store = VectorStoreFactory.load_existing(
            collection_name=vector_config.get("collection_name", "chalmers_courses"),
            persist_directory=persist_dir,
            embedding_model=retrieval_config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        print(f"[OK] Vector store loaded ({vector_store._collection.count()} documents)")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("   Run setup script first: python src/retrieval/langchain_setup.py")
        return
    
    # Create retriever
    top_k = retrieval_config.get("top_k", 3)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Create LLM
    print("\n[3/4] Loading LLM...")
    try:
        llm = create_llm_from_config(config.config)
        print("[OK] LLM loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load LLM: {e}")
        print("\nMake sure you have:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Requested access to the model")
        return
    
    # Create RAG chain
    print("\n[4/4] Building RAG chain...")
    rag_chain = create_rag_chain(
        retriever=retriever,
        llm=llm,
        template_name="rag"
    )
    print("[OK] RAG chain ready")
    
    # Test queries
    test_queries = [
        "What courses teach deep learning?",
        "What are the compulsory courses?",
        "Which courses cover machine learning?",
        "What courses are about optimization?",
        "Show me courses with 7.5 credits"
    ]
    
    print("\n" + "=" * 70)
    print("Testing RAG Pipeline")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 70)
        
        # Retrieve documents
        print("\nRetrieved documents:")
        retrieved_docs = retriever.invoke(query)
        for j, doc in enumerate(retrieved_docs, 1):
            course_code = doc.metadata.get('course_code', 'Unknown')
            course_name = doc.metadata.get('course_name', '')
            print(f"  {j}. {course_code}: {course_name}")
        
        # Generate answer
        print("\nGenerating answer...")
        start_time = time.time()
        
        try:
            result = rag_chain.invoke_with_sources(query)
            generation_time = time.time() - start_time
            
            print("\n" + "-" * 70)
            print("ANSWER:")
            print("-" * 70)
            print(result['answer'])
            print("\n" + "-" * 70)
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Generation time: {generation_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("[OK] RAG pipeline test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

