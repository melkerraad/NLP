"""Quick test script for the retrieval system."""

from pathlib import Path
from src.retrieval.setup_retrieval import CourseRetriever

# Get project root and database directory
project_root = Path(__file__).parent
db_directory = project_root / "data" / "chroma_db"

# Initialize retriever (will use existing collection if it exists)
retriever = CourseRetriever(
    collection_name="chalmers_courses",
    persist_directory=str(db_directory)
)

# Test query
query = "What courses teach deep learning?"
print(f"Query: {query}\n")
print("Results:")
print("-" * 60)

results = retriever.retrieve(query, top_k=5)

if results:
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['course_code']}: {r['course_name']}")
        if r.get('similarity'):
            print(f"   Similarity: {r['similarity']:.3f}")
        print(f"   URL: {r['url']}\n")
else:
    print("No results found.")

