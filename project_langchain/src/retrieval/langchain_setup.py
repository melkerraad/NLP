"""Setup script to create LangChain vector store from course data."""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
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


def parse_course_sections(description: str) -> Dict[str, str]:
    """Parse course description into semantic sections.
    
    Args:
        description: Full course description text
        
    Returns:
        Dictionary mapping section names to section content
    """
    sections = {}
    
    # Define section headers to look for (order matters - more specific first)
    # Use word boundaries to avoid partial matches
    section_patterns = [
        (r'\bCourse specific prerequisites\b', 'prerequisites'),
        (r'\bExamination including compulsory elements\b', 'examination'),
        (r'\bLearning outcomes\s*\([^)]*\)\b', 'learning_outcomes'),
        (r'\bAim\b', 'aim'),
        (r'\bContent\b', 'content'),
        (r'\bOrganisation\b', 'organisation'),
        (r'\bLiterature\b', 'literature'),
    ]
    
    # Find all section boundaries
    section_boundaries = []
    for pattern, section_name in section_patterns:
        matches = list(re.finditer(pattern, description, re.IGNORECASE))
        for match in matches:
            # Use end of match as start of content (after header)
            section_boundaries.append((match.end(), section_name, match.group(0)))
    
    # Sort by position in text
    section_boundaries.sort(key=lambda x: x[0])
    
    # Remove duplicates (keep first occurrence if same section appears multiple times)
    seen_sections = {}
    unique_boundaries = []
    for pos, section_name, header in section_boundaries:
        if section_name not in seen_sections:
            seen_sections[section_name] = True
            unique_boundaries.append((pos, section_name, header))
    section_boundaries = unique_boundaries
    section_boundaries.sort(key=lambda x: x[0])
    
    # Extract sections
    for i, (start_pos, section_name, header) in enumerate(section_boundaries):
        # Find end position (start of next section or end of text)
        if i + 1 < len(section_boundaries):
            end_pos = section_boundaries[i + 1][0]
        else:
            end_pos = len(description)
        
        # Extract section content (from after header to next section)
        section_content = description[start_pos:end_pos].strip()
        
        # Clean up: remove any trailing section headers that might have been included
        for other_pattern, other_name in section_patterns:
            if other_name != section_name:
                # Remove trailing headers
                section_content = re.sub(
                    other_pattern + r'.*$', 
                    '', 
                    section_content, 
                    flags=re.IGNORECASE | re.DOTALL
                ).strip()
        
        if section_content and len(section_content) > 10:  # Minimum content length
            sections[section_name] = section_content
    
    # If no sections found, return entire description as "general" section
    if not sections:
        sections['general'] = description
    
    return sections


def courses_to_documents(courses: List[Dict]) -> List[Document]:
    """Convert course dictionaries to LangChain Documents with semantic chunking.
    
    Each course is split into semantic sections (prerequisites, aim, learning outcomes,
    content, etc.), and each section becomes a separate document.
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        List of LangChain Document objects (one per section per course)
    """
    documents = []
    
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
        
        # Parse description into semantic sections
        if description:
            sections = parse_course_sections(description)
        else:
            sections = {}
        
        # Create a document for each section
        for section_name, section_content in sections.items():
            # Build document text with course context
            doc_parts = []
            
            # Always include course code and name
            if course_code or course_name:
                doc_parts.append(f"Course: {course_code} - {course_name}")
            
            # Include course type
            if course_type:
                doc_parts.append(f"This course is {course_type}.")
            
            # Add section header and content
            section_header_map = {
                'prerequisites': 'Course Specific Prerequisites',
                'aim': 'Aim',
                'learning_outcomes': 'Learning Outcomes',
                'content': 'Content',
                'organisation': 'Organisation',
                'literature': 'Literature',
                'examination': 'Examination including compulsory elements',
                'general': 'Course Information'
            }
            section_header = section_header_map.get(section_name, section_name.title())
            doc_parts.append(f"{section_header}:\n{section_content}")
            
            # Combine into document text
            document_text = "\n\n".join(doc_parts)
            
            # Create metadata with section information
            metadata = base_metadata.copy()
            metadata['section'] = section_name
            
            # Create LangChain Document
            doc = Document(
                page_content=document_text,
                metadata=metadata
            )
            documents.append(doc)
        
        # If no description or sections, create a minimal document with just course info
        if not sections:
            doc_parts = []
            if course_code or course_name:
                doc_parts.append(f"Course: {course_code} - {course_name}")
            if course_type:
                doc_parts.append(f"This course is {course_type}.")
            
            if doc_parts:
                metadata = base_metadata.copy()
                metadata['section'] = 'general'
                doc = Document(
                    page_content="\n\n".join(doc_parts),
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
    
    # Convert to LangChain Documents (with semantic chunking)
    print("\nConverting courses to LangChain Documents with semantic chunking...")
    documents = courses_to_documents(courses)
    print(f"[OK] Created {len(documents)} document chunks from {len(courses)} courses")
    print(f"   Average: {len(documents)/len(courses):.1f} chunks per course")
    
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

