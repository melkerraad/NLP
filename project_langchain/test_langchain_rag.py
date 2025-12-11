"""Test script for LangChain RAG pipeline."""

from pathlib import Path
import sys
import time
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStoreFactory
from src.generation.llm_factory import create_llm_from_config
from src.generation.rag_chain import create_rag_chain
from src.utils.timing import TimingTracker


def main():
    """Run RAG pipeline tests."""
    tracker = TimingTracker()
    
    print("=" * 70)
    print("LangChain RAG Pipeline Test")
    print("=" * 70)
    
    # Check GPU availability
    print("\n[INFO] Device Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    with tracker.time_operation("config_loading"):
        config = load_config()
    print("[OK] Configuration loaded")
    
    # Load vector store
    print("\n[2/4] Loading vector store...")
    vector_config = config.get_vector_store_config()
    retrieval_config = config.get_retrieval_config()
    project_root = Path(__file__).parent
    
    persist_dir = str(project_root / vector_config.get("persist_directory", "data/chroma_db"))
    
    # Auto-detect device for embeddings
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        with tracker.time_operation("vector_store_loading"):
            vector_store = VectorStoreFactory.load_existing(
                collection_name=vector_config.get("collection_name", "chalmers_courses"),
                persist_directory=persist_dir,
                embedding_model=retrieval_config.get("embedding_model", "all-MiniLM-L6-v2"),
                device=embedding_device
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
        with tracker.time_operation("llm_loading"):
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
    with tracker.time_operation("rag_chain_building"):
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
        
        query_start_time = time.time()
        
        # Retrieve documents
        print("\n[Step 1] Retrieving documents...")
        with tracker.time_operation("retrieval"):
            retrieved_docs = retriever.invoke(query)
        
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            course_code = doc.metadata.get('course_code', 'Unknown')
            course_name = doc.metadata.get('course_name', '')
            print(f"  {j}. {course_code}: {course_name}")
        
        # Generate answer
        print("\n[Step 2] Generating answer...")
        try:
            with tracker.time_operation("generation"):
                result = rag_chain.invoke_with_sources(query)
            
            total_query_time = time.time() - query_start_time
            
            print("\n" + "-" * 70)
            print("ANSWER:")
            print("-" * 70)
            print(result['answer'])
            print("\n" + "-" * 70)
            print("PERFORMANCE:")
            print("-" * 70)
            retrieval_time = tracker.get_timing("retrieval")
            generation_time = tracker.get_timing("generation")
            
            if retrieval_time:
                print(f"  Retrieval time: {retrieval_time*1000:.2f} ms")
            if generation_time:
                print(f"  Generation time: {generation_time*1000:.2f} ms")
            print(f"  Total query time: {total_query_time*1000:.2f} ms")
            print(f"  Sources: {', '.join(result['sources'])}")
            
        except Exception as e:
            print(f"[ERROR] Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("[OK] RAG pipeline test completed!")
    print("=" * 70)
    
    # Print timing summary
    tracker.print_summary("Overall Performance Summary")


if __name__ == "__main__":
    main()

