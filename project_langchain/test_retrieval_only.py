"""Test retrieval only (without LLM generation)."""

from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStoreFactory


def main():
    """Test retrieval without LLM."""
    print("=" * 70)
    print("Testing Retrieval Only (No LLM)")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    vector_config = config.get_vector_store_config()
    retrieval_config = config.get_retrieval_config()
    
    # Load vector store
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
        print(f"Query {i}: {query}")
        print("-" * 70)
        
        retrieved_docs = retriever.invoke(query)
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            course_code = doc.metadata.get('course_code', 'Unknown')
            course_name = doc.metadata.get('course_name', '')
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"  {j}. {course_code}: {course_name}")
            print(f"     Preview: {preview}...")
    
    print("\n" + "=" * 70)
    print("[OK] Retrieval test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

