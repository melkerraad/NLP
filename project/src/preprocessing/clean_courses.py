"""Clean and preprocess scraped course data.

Removes empty/null fields and cleans up the data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def remove_empty_fields(course: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields that are None, empty lists, or empty strings.
    
    Also removes credits, prerequisites, and programs fields as they are not needed.
    
    Args:
        course: Course dictionary
        
    Returns:
        Cleaned course dictionary
    """
    cleaned = {}
    
    # Fields to always remove
    fields_to_remove = ['credits', 'prerequisites', 'programs']
    
    for key, value in course.items():
        # Skip fields we don't want
        if key in fields_to_remove:
            continue
        
        # Skip None values
        if value is None:
            continue
        
        # Skip empty lists (except sections which might be empty but are important)
        if isinstance(value, list) and len(value) == 0:
            if key != 'sections':
                continue
        
        # Skip empty strings
        if isinstance(value, str) and value.strip() == "":
            continue
        
        # Clean sections if present
        if key == 'sections' and isinstance(value, list):
            # Remove sections with empty content
            cleaned_sections = []
            for section in value:
                if isinstance(section, dict):
                    content = section.get('content', '')
                    if content and content.strip():
                        cleaned_sections.append(section)
            cleaned[key] = cleaned_sections if cleaned_sections else value
        else:
            # Keep all other values
            cleaned[key] = value
    
    return cleaned


def clean_course_name(name: str) -> str:
    """Clean course name by removing common prefixes.
    
    Args:
        name: Raw course name
        
    Returns:
        Cleaned course name
    """
    if not name:
        return name
    
    # Remove "Course syllabus for" prefix
    name = name.replace("Course syllabus for", "").strip()
    
    return name


def clean_courses(input_path: Path, output_path: Path) -> None:
    """Clean course data by removing empty fields and fixing issues.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output cleaned JSON file
    """
    print(f"Loading courses from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    print(f"Found {len(courses)} courses")
    print("Cleaning courses...")
    
    cleaned_courses = []
    for course in courses:
        # Clean course name
        if 'course_name' in course:
            course['course_name'] = clean_course_name(course['course_name'])
        
        # Remove empty/null fields and unwanted fields (credits, prerequisites, programs)
        cleaned_course = remove_empty_fields(course)
        cleaned_courses.append(cleaned_course)
    
    # Count removed fields
    original_fields = set()
    cleaned_fields = set()
    if courses:
        original_fields = set(courses[0].keys())
        if cleaned_courses:
            cleaned_fields = set(cleaned_courses[0].keys())
    
    removed_fields = original_fields - cleaned_fields
    
    print(f"\nCleaning summary:")
    print(f"  - Removed {len(removed_fields)} empty/null fields: {', '.join(sorted(removed_fields))}")
    print(f"  - Kept {len(cleaned_fields)} fields with data")
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_courses, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(cleaned_courses)} cleaned courses to {output_path}")


def main():
    """Main function."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "raw" / "courses_test.json"
    output_file = project_root / "data" / "processed" / "courses_clean.json"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    clean_courses(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Review the cleaned data in data/processed/courses_clean.json")
    print("2. Use this cleaned data for the retrieval system")
    print()


if __name__ == "__main__":
    main()

