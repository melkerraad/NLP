"""Setup script to create LangChain vector store from course data."""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def extract_program_info(description: str) -> List[str]:
    """Extract program information from course description.
    
    Looks for patterns like "In programmesMPDSC - Data Science and AI, Year 1(compulsory)"
    or "MPDSC - Data Science and AI, Year 1(compulsory)"
    
    Args:
        description: Full course description text
        
    Returns:
        List of program strings (e.g., ["MPDSC - Data Science and AI, Year 1 (compulsory)"])
    """
    programs = []
    
    # Pattern to match program codes and names
    # Format: "MPDSC - Data Science and AI, Year 1(compulsory)" or similar
    # Look for patterns like: CODE - Name, Year X(type)
    pattern = r'([A-Z]{2,6})\s*-\s*([^,]+?),\s*Year\s*\d+\s*\(([^)]+)\)'
    matches = re.finditer(pattern, description, re.IGNORECASE)
    
    for match in matches:
        code = match.group(1)
        name = match.group(2).strip()
        course_type = match.group(3).strip()
        program_str = f"{code} - {name}, Year {match.group(0).split('Year ')[1].split('(')[0].strip()} ({course_type})"
        programs.append(program_str)
    
    # Also try simpler pattern: "In programmes" followed by text until "Examiner"
    if not programs:
        pattern2 = r'In programmes([^E]+?)(?:Examiner|$)'
        matches2 = re.finditer(pattern2, description, re.IGNORECASE | re.DOTALL)
        for match in matches2:
            program_text = match.group(1).strip()
            # Clean up the text
            program_text = re.sub(r'\s+', ' ', program_text)  # Normalize whitespace
            if program_text and len(program_text) > 10:  # Minimum length
                programs.append(program_text)
    
    return programs


def create_metadata_chunk(course: Dict) -> Document:
    """Create a metadata chunk with course information (everything except description).
    
    Args:
        course: Course dictionary
        
    Returns:
        Document with course metadata
    """
    course_code = course.get('course_code', '')
    course_name = course.get('course_name', '')
    course_type = course.get('course_type', '')
    credits = course.get('credits')
    prerequisites = course.get('prerequisites', [])
    description = course.get('description', '')
    
    # Extract program information from description
    programs = extract_program_info(description)
    
    # Build metadata text
    metadata_parts = []
    
    # Course identification
    if course_code or course_name:
        metadata_parts.append(f"Course Code: {course_code}")
        metadata_parts.append(f"Course Name: {course_name}")
    
    # Course type
    if course_type:
        metadata_parts.append(f"Course Type: {course_type}")
    
    # Credits
    if credits:
        metadata_parts.append(f"Credits: {credits}")
    
    # Prerequisites
    if prerequisites:
        prereq_str = ', '.join(prerequisites)
        metadata_parts.append(f"Prerequisites: {prereq_str}")
    
    # Program information (which programs this course is part of)
    if programs:
        metadata_parts.append("\nProgram Information:")
        for program in programs:
            metadata_parts.append(f"  - {program}")
    
    # URL
    url = course.get('url', '')
    if url:
        metadata_parts.append(f"\nCourse URL: {url}")
    
    metadata_text = "\n".join(metadata_parts)
    
    # Create metadata
    metadata = {
        'course_code': course_code,
        'course_name': course_name,
        'course_type': course_type,
        'url': url,
        'chunk_type': 'metadata'
    }
    
    return Document(
        page_content=metadata_text,
        metadata=metadata
    )


def courses_to_documents(courses: List[Dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Convert course dictionaries to LangChain Documents with overlap-based chunking.
    
    Each course creates:
    1. One metadata chunk (course info, programs, prerequisites - everything except description)
    2. Multiple description chunks (description split with overlap)
    
    Args:
        courses: List of course dictionaries
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    # Initialize text splitter for description chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs, sentences, words
    )
    
    for course in courses:
        course_code = course.get('course_code', '')
        course_name = course.get('course_name', '')
        course_type = course.get('course_type', '')
        description = course.get('description', '')
        
        # Base metadata for all chunks from this course
        base_metadata = {
            'course_code': course_code,
            'course_name': course_name,
            'course_type': course_type,
            'url': course.get('url', '')
        }
        
        # 1. Create metadata chunk (everything except description)
        metadata_doc = create_metadata_chunk(course)
        metadata_doc.metadata.update(base_metadata)
        documents.append(metadata_doc)
        
        # 2. Create description chunks with overlap
        if description:
            # Add course context to description
            description_with_context = f"Course: {course_code} - {course_name}\n\n"
            if course_type:
                description_with_context += f"This course is {course_type}.\n\n"
            description_with_context += f"Course Description:\n{description}"
            
            # Split description into chunks
            description_chunks = text_splitter.split_text(description_with_context)
            
            # Create a document for each description chunk
            for i, chunk_text in enumerate(description_chunks):
                metadata = base_metadata.copy()
                metadata['chunk_type'] = 'description'
                metadata['chunk_index'] = i
                metadata['total_chunks'] = len(description_chunks)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                documents.append(doc)
        else:
            # If no description, create a minimal description chunk
            minimal_text = f"Course: {course_code} - {course_name}"
            if course_type:
                minimal_text += f"\nThis course is {course_type}."
            
            metadata = base_metadata.copy()
            metadata['chunk_type'] = 'description'
            metadata['chunk_index'] = 0
            metadata['total_chunks'] = 1
            
            doc = Document(
                page_content=minimal_text,
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
    
    # Convert to LangChain Documents (with overlap-based chunking)
    print("\nConverting courses to LangChain Documents with overlap-based chunking...")
    chunk_size = 500  # Characters per chunk
    chunk_overlap = 100  # Overlap between chunks
    print(f"   Chunk size: {chunk_size} characters, Overlap: {chunk_overlap} characters")
    documents = courses_to_documents(courses, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] Created {len(documents)} document chunks from {len(courses)} courses")
    print(f"   Average: {len(documents)/len(courses):.1f} chunks per course")
    print(f"   (Each course has 1 metadata chunk + multiple description chunks)")
    
    # Create vector store (always overwrite in setup mode)
    print("\nCreating vector store...")
    persist_dir = str(project_root / vector_config.get("persist_directory", "data/chroma_db"))
    
    # Auto-detect device for embeddings
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU optimizations enabled for faster embedding")
        expected_time = f"~{len(documents) * 0.1:.0f}-{len(documents) * 0.3:.0f} seconds"
    else:
        print("[INFO] Using CPU for embeddings")
        expected_time = f"~{len(documents) * 2:.0f}-{len(documents) * 5:.0f} seconds"
    
    print(f"[INFO] Embedding {len(documents)} documents (this may take {expected_time})...")
    
    import time
    start_time = time.time()
    
    vector_store = VectorStoreFactory.create_vector_store(
        collection_name=vector_config.get("collection_name", "chalmers_courses"),
        persist_directory=persist_dir,
        embedding_model=retrieval_config.get("embedding_model", "all-MiniLM-L6-v2"),
        documents=documents,
        device=embedding_device,
        overwrite=True  # Always overwrite in setup mode
    )
    
    embedding_time = time.time() - start_time
    print(f"[OK] Embedding completed in {embedding_time:.1f} seconds")
    if embedding_device == "cuda":
        print(f"[INFO] Performance: {len(documents)/embedding_time:.1f} documents/second")
    
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

