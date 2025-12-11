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


# Note: Program extraction is now done during scraping, so this function is kept for backward compatibility
def extract_program_info(description: str) -> List[str]:
    """Extract program information from course description (legacy function).
    
    This is kept for backward compatibility. New scraping extracts programs directly.
    
    Args:
        description: Full course description text
        
    Returns:
        List of program strings
    """
    programs = []
    
    # Pattern to match program codes and names
    pattern = r'([A-Z]{2,6})\s*-\s*([^,]+?),\s*Year\s*\d+\s*\(([^)]+)\)'
    matches = re.finditer(pattern, description, re.IGNORECASE)
    
    for match in matches:
        code = match.group(1)
        name = match.group(2).strip()
        course_type = match.group(3).strip()
        program_str = f"{code} - {name}, Year {match.group(0).split('Year ')[1].split('(')[0].strip()} ({course_type})"
        programs.append(program_str)
    
    return programs


def create_metadata_chunk(course: Dict) -> Document:
    """Create a metadata chunk with course information (everything except sections).
    
    Args:
        course: Course dictionary
        
    Returns:
        Document with course metadata
    """
    course_code = course.get('course_code', '')
    course_name = course.get('course_name', '')
    course_type = course.get('course_type', '')
    
    # Build metadata text
    metadata_parts = []
    
    # Course identification
    if course_code or course_name:
        metadata_parts.append(f"Course Code: {course_code}")
        metadata_parts.append(f"Course Name: {course_name}")
    
    # Course type
    if course_type:
        metadata_parts.append(f"Course Type: {course_type}")
    
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
    """Convert course dictionaries to LangChain Documents using pre-chunked sections.
    
    Each course creates:
    1. One metadata chunk (course code, name, type, URL)
    2. One document per section (pre-chunked from HTML h2 headers during scraping)
    3. If sections are very long, they may be further split with overlap
    
    Args:
        courses: List of course dictionaries (with 'sections' field from scraping)
        chunk_size: Size of each chunk in characters (for splitting long sections)
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    # Initialize text splitter for splitting very long sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for course in courses:
        course_code = course.get('course_code', '')
        course_name = course.get('course_name', '')
        course_type = course.get('course_type', '')
        sections = course.get('sections', [])
        
        # Base metadata for all chunks from this course
        base_metadata = {
            'course_code': course_code,
            'course_name': course_name,
            'course_type': course_type,
            'url': course.get('url', '')
        }
        
        # 1. Create metadata chunk
        metadata_doc = create_metadata_chunk(course)
        metadata_doc.metadata.update(base_metadata)
        documents.append(metadata_doc)
        
        # 2. Create one document per section (pre-chunked from HTML)
        if sections:
            for section in sections:
                section_name = section.get('section_name', 'unknown')
                section_name_original = section.get('section_name_original', section_name)
                content = section.get('content', '')
                
                if not content or len(content.strip()) < 10:
                    continue  # Skip empty sections
                
                # Build section text with course context
                section_text = f"Course: {course_code} - {course_name}\n"
                if course_type:
                    section_text += f"This course is {course_type}.\n\n"
                section_text += f"{section_name_original}:\n{content}"
                
                # If section is very long, split it further
                if len(section_text) > chunk_size * 1.5:  # If significantly longer than chunk_size
                    section_chunks = text_splitter.split_text(section_text)
                    
                    for i, chunk_text in enumerate(section_chunks):
                        metadata = base_metadata.copy()
                        metadata['chunk_type'] = 'section'
                        metadata['section_name'] = section_name
                        metadata['section_name_original'] = section_name_original
                        metadata['chunk_index'] = i
                        metadata['total_chunks'] = len(section_chunks)
                        
                        doc = Document(
                            page_content=chunk_text,
                            metadata=metadata
                        )
                        documents.append(doc)
                else:
                    # Section fits in one chunk
                    metadata = base_metadata.copy()
                    metadata['chunk_type'] = 'section'
                    metadata['section_name'] = section_name
                    metadata['section_name_original'] = section_name_original
                    
                    doc = Document(
                        page_content=section_text,
                        metadata=metadata
                    )
                    documents.append(doc)
        else:
            # Fallback: if no sections (old data format), create minimal chunk
            minimal_text = f"Course: {course_code} - {course_name}"
            if course_type:
                minimal_text += f"\nThis course is {course_type}."
            
            metadata = base_metadata.copy()
            metadata['chunk_type'] = 'general'
            
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
    courses_file_path = data_config.get("courses_file", "data/processed/courses_clean.json")
    
    # Resolve path - handle both relative and absolute paths
    if Path(courses_file_path).is_absolute():
        courses_file = Path(courses_file_path)
    else:
        # If path starts with ../, resolve from project_root's parent
        if courses_file_path.startswith("../"):
            # Go up from project_langchain to parent directory, then follow the rest of the path
            courses_file = (project_root.parent / courses_file_path).resolve()
        else:
            # Relative to project_root (project_langchain)
            courses_file = (project_root / courses_file_path).resolve()
    
    # Check if courses file exists
    if not courses_file.exists():
        print(f"[ERROR] Courses file not found: {courses_file}")
        print(f"   Expected at: {courses_file}")
        print(f"   Project root: {project_root}")
        print("   Run preprocessing script first!")
        return
    
    # Load courses
    courses = load_courses(courses_file)
    
    # Convert to LangChain Documents (using pre-chunked sections from HTML)
    print("\nConverting courses to LangChain Documents using pre-chunked sections...")
    chunk_size = 500  # Characters per chunk (for splitting very long sections)
    chunk_overlap = 100  # Overlap between chunks
    print(f"   Using sections extracted from HTML h2 headers during scraping")
    print(f"   Long sections (>750 chars) will be further split with {chunk_overlap} char overlap")
    documents = courses_to_documents(courses, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] Created {len(documents)} document chunks from {len(courses)} courses")
    print(f"   Average: {len(documents)/len(courses):.1f} chunks per course")
    print(f"   (Each course has 1 metadata chunk + multiple section chunks)")
    
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

