"""Simple test of Qwen model with RAG."""

from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.retrieval.setup_retrieval import CourseRetriever
from src.generation.llama_generator import LlamaRAGGenerator

print("=" * 70)
print("Testing Qwen 2.5 0.5B RAG System")
print("=" * 70)

# Initialize retriever
print("\n[1/3] Loading retrieval system...")
db_directory = project_root / "data" / "chroma_db"
retriever = CourseRetriever(
    collection_name="chalmers_courses",
    persist_directory=str(db_directory)
)
print(f"Retrieval ready: {retriever.get_collection_size()} courses")

# Initialize generator
print("\n[2/3] Loading Qwen model (this may take a minute)...")
generator = LlamaRAGGenerator(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_alternative_if_gated=False  # Don't try Llama, use Qwen directly
)
print("Generator ready!")

# Test query
print("\n[3/3] Testing RAG...")
query = "What courses teach deep learning?"
print(f"\nQuery: {query}")

# Retrieve
retrieved = retriever.retrieve(query, top_k=3)
print(f"\nRetrieved {len(retrieved)} courses:")
for i, r in enumerate(retrieved, 1):
    print(f"  {i}. {r['course_code']}: {r['course_name']}")

# Format context
context_docs = [
    {
        'course_code': r['course_code'],
        'course_name': r['course_name'],
        'document': r['document']
    }
    for r in retrieved
]

# Generate (with shorter output for testing)
print("\nGenerating response (this may take 30-60 seconds on CPU)...")
try:
    result = generator.generate(
        query, 
        context_docs,
        max_new_tokens=200,  # Shorter for testing
        temperature=0.7
    )
    
    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(result)
    print("\n" + "=" * 70)
    print("SUCCESS! RAG system is working!")
    print("=" * 70)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

