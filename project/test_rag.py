"""Simple RAG proof of concept test with Llama 3.2 1B.

This script demonstrates the RAG pipeline:
1. Loads the vector database (already created from scraped Chalmers courses)
2. Retrieves relevant courses for a query
3. Generates a response using Llama 3.2 1B (~2GB, smaller than 3B)
"""

from pathlib import Path
import sys
import time
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent

# Load environment variables from .env file (explicitly from project root)
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)
sys.path.insert(0, str(project_root))

from src.retrieval.setup_retrieval import CourseRetriever
from src.generation.llama_generator import LlamaRAGGenerator


def main():
    """Run RAG proof of concept test."""
    print("=" * 70)
    print("RAG Pipeline Proof of Concept - Llama 3.2 1B")
    print("=" * 70)
    
    # Initialize retriever
    print("\n[1/2] Loading vector database...")
    db_directory = project_root / "data" / "chroma_db"
    retriever = CourseRetriever(
        collection_name="chalmers_courses",
        persist_directory=str(db_directory)
    )
    print(f"[OK] Vector database loaded ({retriever.get_collection_size()} courses)")
    
    # Initialize generator
    print("\n[2/2] Loading Llama 3.2 1B model...")
    print("      (Using 1B model - ~2GB instead of 3B - ~6GB)")
    print("      (This may take a few minutes on first run)")
    try:
        generator = LlamaRAGGenerator(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            use_alternative_if_gated=False  # Fail clearly if no access
        )
        print("[OK] Llama 3.2 1B model loaded")
        
        # Warmup generation to compile model (if using torch.compile)
        print("\n[Warmup] Running initial generation to compile model...")
        try:
            dummy_context = [{'course_code': 'TEST', 'course_name': 'Test Course', 'document': 'Test content'}]
            generator.generate_with_sources("test", dummy_context, max_new_tokens=10)
            print("[OK] Warmup complete")
        except Exception as e:
            print(f"[WARNING] Warmup failed: {e}")
            print("   Continuing anyway...")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("\nMake sure you have:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Requested access to Llama 3.2 at:")
        print("     https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        return
    
    # Test queries
    test_queries = [
        "What courses teach deep learning?",
        "What are the compulsory courses?",
        "Which courses cover machine learning?",
        "What courses are about optimization?",
        "Show me courses with 7.5 credits"
    ]
    
    print("\n" + "=" * 70)
    print("Testing RAG Pipeline with Multiple Queries")
    print("=" * 70)
    
    # Process each query
    for query_num, query in enumerate(test_queries, 1):
        print(f"\n\n{'=' * 70}")
        print(f"Query {query_num}/{len(test_queries)}: {query}")
        print("-" * 70)
        
        # Retrieve relevant courses
        retrieved = retriever.retrieve(query, top_k=3)
        
        if not retrieved:
            print("[ERROR] No courses retrieved!")
            continue
        
        print(f"\nRetrieved {len(retrieved)} courses:")
        for i, r in enumerate(retrieved, 1):
            print(f"   {i}. {r['course_code']}: {r['course_name']}")
        
        # Format context for generator
        context_docs = [
            {
                'course_code': r['course_code'],
                'course_name': r['course_name'],
                'document': r['document']
            }
            for r in retrieved
        ]
        
        # Generate response
        print("\nGenerating response...")
        try:
            start_time = time.time()
            result = generator.generate_with_sources(query, context_docs)
            generation_time = time.time() - start_time
            
            print("\n" + "-" * 70)
            print("ANSWER:")
            print("-" * 70)
            print(result['answer'])
            print("\n" + "-" * 70)
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Generation time: {generation_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] Error generating response: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("[OK] RAG pipeline test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

