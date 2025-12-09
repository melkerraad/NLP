"""Test the full RAG pipeline with Llama 3.2 3B."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.retrieval.setup_retrieval import CourseRetriever
from src.generation.llama_generator import LlamaRAGGenerator


def main():
    """Test the complete RAG system."""
    print("=" * 70)
    print("Full RAG Pipeline Test - Llama 3.2 3B")
    print("=" * 70)
    
    # Initialize retriever
    print("\n[1/3] Loading retrieval system...")
    db_directory = project_root / "data" / "chroma_db"
    retriever = CourseRetriever(
        collection_name="chalmers_courses",
        persist_directory=str(db_directory)
    )
    print(f"✅ Retrieval system ready ({retriever.get_collection_size()} courses indexed)")
    
    # Initialize generator
    print("\n[2/3] Loading Llama 3.2 3B model...")
    print("      (This may take a few minutes on first run)")
    try:
        generator = LlamaRAGGenerator()
        print("✅ Generator ready")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have:")
        print("  1. Set HF_TOKEN environment variable")
        print("     Get token at: https://huggingface.co/settings/tokens")
        print("  2. Installed required packages: pip install transformers accelerate")
        print("  3. Have GPU access (or it will use CPU, which is slower)")
        return
    
    # Test queries
    test_queries = [
        "What courses teach deep learning?",
        "Which courses cover natural language processing?",
        "What are some machine learning courses?",
    ]
    
    print("\n[3/3] Testing RAG pipeline...")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print("-" * 70)
        
        # Retrieve relevant courses
        retrieved = retriever.retrieve(query, top_k=3)
        
        if not retrieved:
            print("❌ No courses retrieved!")
            continue
        
        print(f"\nRetrieved {len(retrieved)} courses:")
        for j, r in enumerate(retrieved, 1):
            print(f"  {j}. {r['course_code']}: {r['course_name']}")
        
        # Format for generator
        context_docs = []
        for r in retrieved:
            context_docs.append({
                'course_code': r['course_code'],
                'course_name': r['course_name'],
                'document': r['document']
            })
        
        # Generate response
        print("\nGenerating response...")
        try:
            result = generator.generate_with_sources(query, context_docs)
            
            print(f"\n📝 Answer:")
            print(f"   {result['answer']}")
            print(f"\n📚 Sources: {', '.join(result['sources'])}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

