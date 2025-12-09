"""Example course data collection script.

This is a template to help you get started with data collection.
Modify this based on your actual data sources.
"""

from pathlib import Path
import json
from typing import List, Dict


def collect_course_data() -> List[Dict]:
    """Collect course data from sources.
    
    Returns:
        List of course dictionaries with fields:
        - course_code: str
        - course_name: str
        - credits: float
        - description: str
        - prerequisites: List[str]
        - etc.
    """
    courses = []
    
    # TODO: Implement your data collection logic here
    # Examples:
    # - Web scraping from university website
    # - API calls to course catalog
    # - Reading from CSV/JSON files
    # - Manual data entry
    
    # Example structure:
    example_course = {
        "course_code": "DAT450",
        "course_name": "Natural Language Processing",
        "credits": 7.5,
        "description": "Introduction to NLP concepts and techniques...",
        "prerequisites": ["DAT250", "DAT260"],
        "learning_outcomes": [
            "Understand NLP fundamentals",
            "Apply NLP techniques to real problems"
        ],
        "schedule": "Autumn 2025",
        "instructor": "Dr. X",
        "department": "Computer Science",
        "level": "Master's"
    }
    
    courses.append(example_course)
    
    return courses


def save_courses(courses: List[Dict], output_path: Path) -> None:
    """Save collected courses to JSON file.
    
    Args:
        courses: List of course dictionaries
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(courses)} courses to {output_path}")


def main():
    """Main function to run data collection."""
    # Collect course data
    courses = collect_course_data()
    
    # Save to data/raw directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_path = output_dir / "courses_raw.json"
    
    save_courses(courses, output_path)


if __name__ == "__main__":
    main()

