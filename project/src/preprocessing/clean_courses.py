"""Clean and preprocess scraped course data.

Removes empty/null fields and cleans up the data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def remove_empty_fields(course: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields that are None, empty lists, or empty strings.
    
    Args:
        course: Course dictionary
        
    Returns:
        Cleaned course dictionary
    """
    cleaned = {}
    
    for key, value in course.items():
        # Skip None values
        if value is None:
            continue
        
        # Skip empty lists
        if isinstance(value, list) and len(value) == 0:
            continue
        
        # Skip empty strings
        if isinstance(value, str) and value.strip() == "":
            continue
        
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


def fix_credits(credits: Any, course_code: str) -> Any:
    """Try to fix credits if they look wrong.
    
    If credits is a large number (likely course code), return None.
    Credits should typically be between 0.5 and 15 for Chalmers courses.
    
    Args:
        credits: Credits value (could be wrong)
        course_code: Course code for reference
        
    Returns:
        Fixed credits or None if can't determine
    """
    if credits is None:
        return None
    
    # If credits is a float/int
    if isinstance(credits, (int, float)):
        # If it's a reasonable credit value (0.5 to 30), keep it
        if 0.5 <= credits <= 30:
            return credits
        # Otherwise, it's probably wrong (likely course code number)
        return None
    
    return credits


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
        
        # Fix credits
        if 'credits' in course:
            course['credits'] = fix_credits(course['credits'], course.get('course_code', ''))
        
        # Remove empty/null fields
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

