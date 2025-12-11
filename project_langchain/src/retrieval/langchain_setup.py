"""Setup script to create LangChain vector store from course data."""

import json
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
import torch

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import VectorStoreFactory
from src.utils.config_loader import load_config


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


def courses_to_documents(courses: List[Dict]) -> List[Document]:
    """Convert course dictionaries to LangChain Documents.
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for course in courses:
        # Create combined text representation
        course_text_parts = []
        
        # Add course code and name
        course_code = course.get('course_code', '')
        course_name = course.get('course_name', '')
        if course_code or course_name:
            course_text_parts.append(f"Course: {course_code} - {course_name}")
        
        # Add course type if available
        if course.get('course_type'):
            course_text_parts.append(f"Type: {course['course_type']}")
        
        # Add description (main content)
        if course.get('description'):
            course_text_parts.append(course['description'])
        
        # Combine into single document text
        document_text = "\n\n".join(course_text_parts)
        
        # Create metadata
        metadata = {
            'course_code': course_code,
            'course_name': course_name,
            'course_type': course.get('course_type', ''),
            'url': course.get('url', '')
        }
        
        # Create LangChain Document
        doc = Document(
            page_content=document_text,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def main():
    """Main function to set up vector store."""
    print("=" * 70)
    print("LangChain Vector Store Setup")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    data_config = config.get_data_config()
    vector_config = config.get_vector_store_config()
    retrieval_config = config.get_retrieval_config()
    
    # Get paths
    project_root = Path(__file__).parent.parent.parent
    courses_file = project_root / data_config.get("courses_file", "data/processed/courses_clean.json")
    
    # Check if courses file exists
    if not courses_file.exists():
        print(f"[ERROR] Courses file not found: {courses_file}")
        print("   Run preprocessing script first!")
        return
    
    # Load courses
    courses = load_courses(courses_file)
    
    # Convert to LangChain Documents
    print("\nConverting courses to LangChain Documents...")
    documents = courses_to_documents(courses)
    print(f"[OK] Created {len(documents)} documents")
    
    # Create vector store
    print("\nCreating vector store...")
    persist_dir = str(project_root / vector_config.get("persist_directory", "data/chroma_db"))
    
    # Auto-detect device for embeddings
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU for embeddings")
    
    vector_store = VectorStoreFactory.create_vector_store(
        collection_name=vector_config.get("collection_name", "chalmers_courses"),
        persist_directory=persist_dir,
        embedding_model=retrieval_config.get("embedding_model", "all-MiniLM-L6-v2"),
        documents=documents,
        device=embedding_device
    )
    
    # Verify
    count = vector_store._collection.count()
    print(f"\n[OK] Vector store setup complete!")
    print(f"   Collection: {vector_config.get('collection_name', 'chalmers_courses')}")
    print(f"   Documents: {count}")
    print(f"   Location: {persist_dir}")
    
    # Test retrieval
    print("\n" + "-" * 70)
    print("Testing retrieval...")
    print("-" * 70)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    test_query = "What courses teach deep learning?"
    results = retriever.invoke(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        course_code = doc.metadata.get('course_code', 'Unknown')
        course_name = doc.metadata.get('course_name', '')
        print(f"  {i}. {course_code}: {course_name}")
        print(f"     Preview: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()

