"""Set up retrieval system for course documents.

This script:
1. Loads cleaned course data
2. Creates embeddings for all courses
3. Stores them in ChromaDB vector database
4. Tests retrieval with sample queries
"""

import json
from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer


class CourseRetriever:
    """Retrieval system for course documents."""
    
    def __init__(self, collection_name: str = "chalmers_courses", persist_directory: str = None):
        """Initialize the retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data (None = in-memory)
        """
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"[OK] Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"[OK] Created new collection: {collection_name}")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Embedding model loaded")
    
    def prepare_documents(self, courses: List[Dict]) -> tuple:
        """Prepare documents for embedding.
        
        Creates a combined text representation of each course.
        
        Args:
            courses: List of course dictionaries
            
        Returns:
            Tuple of (documents, metadata, ids)
        """
        documents = []
        metadata = []
        ids = []
        
        for course in courses:
            # Create a combined text representation
            course_text_parts = []
            
            # Add course code and name
            course_text_parts.append(f"Course: {course.get('course_code', '')} - {course.get('course_name', '')}")
            
            # Add course type if available
            if course.get('course_type'):
                course_text_parts.append(f"Type: {course['course_type']}")
            
            # Add description (main content)
            if course.get('description'):
                course_text_parts.append(course['description'])
            
            # Combine into single document
            document_text = "\n\n".join(course_text_parts)
            documents.append(document_text)
            
            # Store metadata
            metadata.append({
                'course_code': course.get('course_code', ''),
                'course_name': course.get('course_name', ''),
                'course_type': course.get('course_type', ''),
                'url': course.get('url', '')
            })
            
            # Use course code as ID
            ids.append(course.get('course_code', ''))
        
        return documents, metadata, ids
    
    def add_courses(self, courses: List[Dict], batch_size: int = 10):
        """Add courses to the vector database.
        
        Args:
            courses: List of course dictionaries
            batch_size: Number of courses to process at once
        """
        print(f"\nPreparing {len(courses)} courses for embedding...")
        documents, metadata, ids = self.prepare_documents(courses)
        
        print(f"Generating embeddings (this may take a minute)...")
        
        # Process in batches to show progress
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
            batch_num = (i // batch_size) + 1
            print(f"  Processed batch {batch_num}/{total_batches} ({len(batch_docs)} courses)")
        
        print(f"\n[OK] Added {len(courses)} courses to vector database")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant courses for a query.
        
        Args:
            query: User query string
            top_k: Number of courses to retrieve
            
        Returns:
            List of retrieved courses with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        retrieved_courses = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved_courses.append({
                    'course_code': results['metadatas'][0][i].get('course_code', ''),
                    'course_name': results['metadatas'][0][i].get('course_name', ''),
                    'course_type': results['metadatas'][0][i].get('course_type', ''),
                    'url': results['metadatas'][0][i].get('url', ''),
                    'document': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'similarity': 1 - results['distances'][0][i] if 'distances' in results and results['distances'][0][i] else None
                })
        
        return retrieved_courses
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()


def load_courses(data_path: Path) -> List[Dict]:
    """Load cleaned course data from JSON file.
    
    Args:
        data_path: Path to cleaned courses JSON file
        
    Returns:
        List of course dictionaries
    """
    print(f"Loading courses from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    print(f"[OK] Loaded {len(courses)} courses")
    return courses


def test_retrieval(retriever: CourseRetriever):
    """Test the retrieval system with sample queries.
    
    Args:
        retriever: Initialized CourseRetriever instance
    """
    test_queries = [
        "What courses cover machine learning?",
        "Which courses are about natural language processing?",
        "What are the compulsory courses?",
        "Courses about optimization",
        "Computer vision courses"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Retrieval System")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 60)
        
        results = retriever.retrieve(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['course_code']}: {result['course_name']}")
                print(f"   Type: {result['course_type']}")
                if result.get('similarity'):
                    print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   URL: {result['url']}")
        else:
            print("  No results found")


def main():
    """Main function to set up retrieval system."""
    print("=" * 60)
    print("Phase 2: Setting Up Retrieval System")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    courses_file = project_root / "data" / "processed" / "courses_clean.json"
    db_directory = project_root / "data" / "chroma_db"
    
    # Load courses
    if not courses_file.exists():
        print(f"[ERROR] Courses file not found: {courses_file}")
        print("   Run the preprocessing script first!")
        return
    
    courses = load_courses(courses_file)
    
    # Initialize retriever
    print("\n" + "=" * 60)
    print("Step 1: Initializing Vector Database")
    print("=" * 60)
    retriever = CourseRetriever(
        collection_name="chalmers_courses",
        persist_directory=str(db_directory)
    )
    
    # Check if collection already has data
    collection_size = retriever.get_collection_size()
    
    if collection_size > 0:
        print(f"\n⚠️  Collection already contains {collection_size} documents")
        response = input("   Rebuild? (y/n): ").strip().lower()
        if response == 'y':
            # Delete and recreate collection
            retriever.client.delete_collection("chalmers_courses")
            retriever.collection = retriever.client.create_collection("chalmers_courses")
            print("[OK] Collection recreated")
        else:
            print("[OK] Using existing collection")
            test_retrieval(retriever)
            return
    
    # Add courses to vector database
    print("\n" + "=" * 60)
    print("Step 2: Creating Embeddings and Adding to Database")
    print("=" * 60)
    retriever.add_courses(courses)
    
    # Test retrieval
    print("\n" + "=" * 60)
    print("Step 3: Testing Retrieval")
    print("=" * 60)
    test_retrieval(retriever)
    
    print("\n" + "=" * 60)
    print("[OK] Retrieval System Setup Complete!")
    print("=" * 60)
    print(f"\nVector database saved to: {db_directory}")
    print(f"Total courses indexed: {retriever.get_collection_size()}")
    print("\nNext steps:")
    print("1. Test with your own queries")
    print("2. Integrate with generation system (Phase 3)")
    print()


if __name__ == "__main__":
    main()

