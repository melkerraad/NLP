"""Test retrieval only (without LLM generation)."""

from pathlib import Path
import sys
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStoreFactory
from src.utils.timing import TimingTracker


def main():
    """Test retrieval without LLM."""
    tracker = TimingTracker()
    
    print("=" * 70)
    print("Testing Retrieval Only (No LLM)")
    print("=" * 70)
    
    # Check GPU availability
    print("\n[INFO] Device Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load configuration
    print("\n[1/2] Loading configuration...")
    with tracker.time_operation("config_loading"):
        config = load_config()
    vector_config = config.get_vector_store_config()
    retrieval_config = config.get_retrieval_config()
    
    # Load vector store
    print("\n[2/2] Loading vector store...")
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
        return
    
    # Create retriever
    top_k = retrieval_config.get("top_k", 3)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Test queries
    test_queries = [
        "What courses teach deep learning?",
        "What are the compulsory courses?",
        "Which courses cover machine learning?",
    ]
    
    print("\n" + "=" * 70)
    print("Testing Retrieval")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 70)
        
        with tracker.time_operation("retrieval"):
            retrieved_docs = retriever.invoke(query)
        
        retrieval_time = tracker.get_timing("retrieval")
        
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            course_code = doc.metadata.get('course_code', 'Unknown')
            course_name = doc.metadata.get('course_name', '')
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"  {j}. {course_code}: {course_name}")
            print(f"     Preview: {preview}...")
        
        if retrieval_time:
            print(f"\n[PERFORMANCE] Retrieval time: {retrieval_time*1000:.2f} ms")
    
    print("\n" + "=" * 70)
    print("[OK] Retrieval test completed!")
    print("=" * 70)
    
    # Print timing summary
    tracker.print_summary("Retrieval Performance Summary")


if __name__ == "__main__":
    main()


