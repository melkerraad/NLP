"""Chalmers course scraper.

This script scrapes course information from Chalmers University.
Start with a small test before scaling up.

Robots.txt Compliance:
- Chalmers robots.txt allows scraping (Allow: /)
- Only search endpoints are disallowed (/sok/, /en/search/)
- Course pages are allowed to scrape
- Be respectful: use delays, proper User-Agent, rate limiting

Usage:
    python src/data_collection/chalmers_scraper.py
"""

from pathlib import Path
import json
import csv
import time
from typing import List, Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import re


def extract_sections_by_headers(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract course sections based on h2 headers.
    
    Finds all h2 headers and extracts content until the next h2 or end of document.
    This creates semantic chunks at the scraping stage.
    
    Args:
        soup: BeautifulSoup object of the course page
        
    Returns:
        List of dictionaries with 'section_name' and 'content' keys
    """
    sections = []
    
    # Find all h2 headers (these mark semantic sections)
    h2_headers = soup.find_all('h2')
    
    if not h2_headers:
        # Fallback: try h3 or other headers
        h2_headers = soup.find_all(['h2', 'h3'])
    
    if not h2_headers:
        # If no headers found, return empty sections
        return sections
    
    # Extract content for each section
    for i, header in enumerate(h2_headers):
        section_name = header.get_text(strip=True)
        
        # Skip empty or very short headers
        if not section_name or len(section_name) < 2:
            continue
        
        # Find the parent container (usually a div or section)
        # Then find all content between this header and the next header
        parent = header.parent
        
        # Get all siblings after this header until we hit the next header
        content_elements = []
        current = header.next_sibling
        
        # Collect all elements until next h2/h3
        while current:
            # Stop if we hit another header
            if hasattr(current, 'name') and current.name in ['h2', 'h3']:
                break
            
            # Collect this element
            if current:
                content_elements.append(current)
            
            current = current.next_sibling if hasattr(current, 'next_sibling') else None
        
        # Extract text from all collected elements
        content_parts = []
        for elem in content_elements:
            if hasattr(elem, 'get_text'):
                text = elem.get_text(separator=' ', strip=True)
                if text:
                    content_parts.append(text)
            elif isinstance(elem, str):
                text = elem.strip()
                if text:
                    content_parts.append(text)
        
        # Combine content
        content = ' '.join(content_parts).strip()
        
        # If we didn't get much content from siblings, try getting all text from parent
        # and then extract the part after this header
        if len(content) < 50 and parent:
            parent_text = parent.get_text(separator=' ', strip=True)
            # Find the position of this header's text in parent
            header_pos = parent_text.find(section_name)
            if header_pos != -1:
                # Get text after header
                after_header = parent_text[header_pos + len(section_name):].strip()
                # Find next header in parent text (if any)
                if i + 1 < len(h2_headers):
                    next_header = h2_headers[i + 1].get_text(strip=True)
                    next_pos = after_header.find(next_header)
                    if next_pos != -1:
                        after_header = after_header[:next_pos].strip()
                content = after_header
        
        # Only add if there's substantial content
        if content and len(content) > 10:
            # Normalize section name (lowercase, remove special chars)
            normalized_name = section_name.lower().strip()
            
            sections.append({
                'section_name': normalized_name,
                'section_name_original': section_name,
                'content': content
            })
    
    return sections


def extract_program_info_from_sections(sections: List[Dict]) -> List[str]:
    """Extract program information from course sections.
    
    Looks for sections that contain program information (usually in a specific section
    or in the "In programmes" text).
    
    Args:
        sections: List of section dictionaries
        
    Returns:
        List of program strings
    """
    programs = []
    
    # Look for program information in sections
    for section in sections:
        content = section['content']
        section_name = section['section_name'].lower()
        
        # Check if this section mentions programs
        if 'programme' in section_name or 'program' in section_name:
            # Extract program patterns
            # Format: "MPDSC - Data Science and AI, Year 1(compulsory)"
            pattern = r'([A-Z]{2,6})\s*-\s*([^,]+?),\s*Year\s*\d+\s*\(([^)]+)\)'
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                code = match.group(1)
                name = match.group(2).strip()
                course_type = match.group(3).strip()
                # Extract year
                year_match = re.search(r'Year\s*(\d+)', match.group(0))
                year = year_match.group(1) if year_match else ""
                
                program_str = f"{code} - {name}, Year {year} ({course_type})"
                programs.append(program_str)
        
        # Also check content for "In programmes" pattern
        if 'in programmes' in content.lower():
            # Extract everything after "In programmes" until next section marker
            pattern = r'in programmes\s*([^E]+?)(?:examiner|$)'
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                program_text = match.group(1).strip()
                # Clean up
                program_text = re.sub(r'\s+', ' ', program_text)
                if program_text and len(program_text) > 10:
                    # Try to parse individual programs
                    # Look for patterns like "CODE - Name, Year X(type)"
                    program_pattern = r'([A-Z]{2,6})\s*-\s*([^,]+?),\s*Year\s*\d+\s*\(([^)]+)\)'
                    program_matches = re.finditer(program_pattern, program_text, re.IGNORECASE)
                    for pm in program_matches:
                        code = pm.group(1)
                        name = pm.group(2).strip()
                        course_type = pm.group(3).strip()
                        year_match = re.search(r'Year\s*(\d+)', pm.group(0))
                        year = year_match.group(1) if year_match else ""
                        program_str = f"{code} - {name}, Year {year} ({course_type})"
                        if program_str not in programs:
                            programs.append(program_str)
    
    return programs


def scrape_course_page(course_url: str, course_code: str = None, course_type: str = None) -> Optional[Dict]:
    """Scrape a single course page from Chalmers.
    
    Args:
        course_url: URL to the course page
        course_code: Course code (if known, otherwise extracted from page)
        course_type: "compulsory" or "elective" (if known from programme page)
        
    Returns:
        Course dictionary or None if scraping fails
    """
    headers = get_headers()
    
    try:
        response = requests.get(course_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract course code (use provided or extract from page/URL)
        if not course_code:
            # Try to extract from URL
            match = re.search(r'/course-syllabus/([A-Z0-9]+)', course_url)
            if match:
                course_code = match.group(1)
            else:
                # Try to find in page
                code_elem = soup.find(text=re.compile(r'^[A-Z]{3}\d{3}'))
                if code_elem:
                    course_code = code_elem.strip()
                else:
                    course_code = "UNKNOWN"
        
        # Extract course name - look for common patterns
        course_name = None
        # Try h1 or h2 tags
        for tag in ['h1', 'h2']:
            name_elem = soup.find(tag)
            if name_elem:
                text = name_elem.get_text(strip=True)
                # Skip if it's just the course code
                if text and text != course_code and len(text) > 5:
                    course_name = text
                    break
        
        # Extract sections based on h2 headers (semantic chunking at scraping stage)
        sections = extract_sections_by_headers(soup)
        
        # Extract credits - look for patterns like "7.5 credits" or "7,5 hp"
        credits = None
        credits_pattern = re.compile(r'(\d+[.,]\d+|\d+)\s*(?:credits|hp|ECTS)', re.IGNORECASE)
        credits_match = credits_pattern.search(soup.get_text())
        if credits_match:
            try:
                credits_str = credits_match.group(1).replace(',', '.')
                credits = float(credits_str)
            except:
                pass
        
        # Extract prerequisites from sections
        prerequisites = []
        for section in sections:
            if section['section_name'].lower() in ['prerequisites', 'course specific prerequisites']:
                # Look for course codes in the prerequisites section
                course_codes = re.findall(r'\b([A-Z]{3}\d{3})\b', section['content'])
                prerequisites.extend(course_codes)
        
        # Remove duplicates
        prerequisites = list(set(prerequisites))
        
        # Extract program information from sections
        programs = extract_program_info_from_sections(sections)
        
        # Build course dictionary with sections
        course = {
            "course_code": course_code,
            "course_name": course_name or "UNKNOWN",
            "credits": credits,
            "prerequisites": prerequisites,
            "sections": sections,  # Pre-chunked sections from HTML
            "programs": programs,  # Program information
            "course_type": course_type,  # compulsory/elective from programme page
            "url": course_url,
            "scraped_date": time.strftime("%Y-%m-%d")
        }
        
        return course
        
    except Exception as e:
        print(f"Error scraping {course_url}: {e}")
        return None


def get_headers() -> Dict[str, str]:
    """Get respectful headers for requests.
    
    Returns:
        Dictionary of HTTP headers
    """
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Educational Research Project - DAT450 NLP Course)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }


def extract_course_links_from_programme(programme_url: str) -> List[Tuple[str, str, str]]:
    """Extract all course links from a master's programme page.
    
    Args:
        programme_url: URL to the master's programme page
        
    Returns:
        List of tuples: (course_code, course_url, course_type)
        course_type is "compulsory" or "elective"
    """
    headers = get_headers()
    
    try:
        print(f"🔍 Scraping programme page: {programme_url}")
        response = requests.get(programme_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Strategy 1: Find all links to course syllabi (most reliable)
        print("  Searching for course syllabus links...")
        # Find all <a> tags first, then filter
        all_links = soup.find_all('a', href=True)
        course_syllabus_links = [link for link in all_links if '/course-syllabus/' in link.get('href', '')]
        
        if course_syllabus_links:
            print(f"  Found {len(course_syllabus_links)} course syllabus links")
            course_links = []
            seen_codes = set()  # Avoid duplicates
            
            for link in course_syllabus_links:
                href = link.get('href', '')
                course_code = href.split('/course-syllabus/')[-1].rstrip('/')
                
                # Skip duplicates
                if course_code in seen_codes:
                    continue
                seen_codes.add(course_code)
                
                # Build full URL
                if href.startswith('/'):
                    full_url = f"https://www.chalmers.se{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = f"https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/{course_code}/"
                
                # Determine if compulsory or elective from parent/li text
                parent = link.parent
                item_text = ""
                if parent:
                    # Check if parent is an <li>, if so use its text
                    if parent.name == 'li':
                        item_text = parent.get_text().lower()
                    else:
                        # Look for nearest <li> ancestor
                        li_ancestor = parent.find_parent('li')
                        if li_ancestor:
                            item_text = li_ancestor.get_text().lower()
                        else:
                            item_text = parent.get_text().lower()
                
                course_type = "compulsory" if "compulsory" in item_text else "elective"
                course_links.append((course_code, full_url, course_type))
            
            if course_links:
                print(f"  Successfully extracted {len(course_links)} unique courses")
                return course_links
            else:
                print("  Found links but couldn't extract course codes")
        
        # Strategy 2: Try to find the <ul> list (fallback)
        print("  Trying to find course list <ul> element...")
        # Try different class combinations
        course_list = None
        for selector in [
            {'class': 'grid'},
            {'class': re.compile(r'no-list-formatting')},
            {'class': lambda x: x and 'grid' in x and 'no-list-formatting' in x},
        ]:
            course_list = soup.find('ul', selector)
            if course_list:
                break
        
        if course_list:
            print(f"  Found course list with {len(course_list.find_all('li'))} items")
            course_links = []
            list_items = course_list.find_all('li')
            
            for item in list_items:
                # Try to find any link in the <li>
                link = item.find('a')
                if not link:
                    continue
                
                href = link.get('href', '')
                if not href:
                    continue
                
                # Check if it's a course syllabus link
                if '/course-syllabus/' in href:
                    course_code = href.split('/course-syllabus/')[-1].rstrip('/')
                    
                    # Build full URL
                    if href.startswith('/'):
                        full_url = f"https://www.chalmers.se{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = f"https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/{course_code}/"
                    
                    # Determine if compulsory or elective
                    item_text = item.get_text().lower()
                    course_type = "compulsory" if "compulsory" in item_text else "elective"
                    course_links.append((course_code, full_url, course_type))
                else:
                    # Debug: print what links we're finding that don't match
                    if 'course' in href.lower() or 'syllabus' in href.lower():
                        print(f"    Found non-matching link: {href[:80]}")
            
            if course_links:
                print(f"  Successfully extracted {len(course_links)} course links from list")
                return course_links
            else:
                print(f"  Found list but no course syllabus links in it")
        
        # If we get here, nothing worked
        print("⚠️  Could not find course links using any method")
        return []
        
        course_links = []
        list_items = course_list.find_all('li') if course_list else []
        
        for item in list_items:
            link = item.find('a', href=re.compile(r'/course-syllabus/'))
            if link:
                href = link.get('href', '')
                course_code = href.split('/course-syllabus/')[-1].rstrip('/')
                
                # Build full URL
                if href.startswith('/'):
                    full_url = f"https://www.chalmers.se{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = f"https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/{course_code}/"
                
                # Determine if compulsory or elective
                item_text = item.get_text().lower()
                course_type = "compulsory" if "compulsory" in item_text else "elective"
                
                course_links.append((course_code, full_url, course_type))
        
        print(f"✅ Found {len(course_links)} courses on programme page")
        return course_links
        
    except Exception as e:
        print(f"❌ Error extracting course links from {programme_url}: {e}")
        return []


def load_course_links_from_csv(csv_path: Path) -> List[Tuple[str, str, str]]:
    """Load course links from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of tuples: (course_code, course_url, course_type)
    """
    course_links = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                course_links.append((
                    row['course_code'],
                    row['course_url'],
                    row['course_type']
                ))
        print(f"✅ Loaded {len(course_links)} course links from CSV")
        return course_links
    except Exception as e:
        print(f"⚠️  Could not load CSV: {e}")
        return []


def find_course_urls(programme_url: str) -> List[Tuple[str, str, str]]:
    """Find all course URLs from a Chalmers master's programme page.
    
    First tries to load from CSV file if it exists, otherwise scrapes the page.
    
    Args:
        programme_url: URL to the master's programme page
        
    Returns:
        List of tuples: (course_code, course_url, course_type)
    """
    # Check if CSV file exists
    csv_path = get_output_path("course_links", "csv")
    if csv_path.exists():
        print(f"📋 Found existing course_links.csv, using it...")
        course_links = load_course_links_from_csv(csv_path)
        if course_links:
            return course_links
        print("  CSV file empty or invalid, trying to scrape programme page...")
    
    # Fallback to scraping
    return extract_course_links_from_programme(programme_url)


def collect_courses(programme_url: str = None, test_mode: bool = True, max_courses: int = None) -> List[Dict]:
    """Collect course data from Chalmers.
    
    Args:
        programme_url: URL to master's programme page (e.g., Data Science and AI MSc)
        test_mode: If True, scrape only first 5 courses. If False, scrape all.
        max_courses: Maximum number of courses to scrape (for testing)
        
    Returns:
        List of course dictionaries
    """
    courses = []
    
    # Default programme URL if not provided
    if not programme_url:
        programme_url = "https://www.chalmers.se/en/education/find-masters-programme/data-science-and-ai-msc/"
    
    # Step 1: Extract course links from programme page
    print("=" * 60)
    print("Step 1: Extracting course links from programme page")
    print("=" * 60)
    course_links = find_course_urls(programme_url)
    
    if not course_links:
        print("❌ No course links found!")
        print("Please check the programme URL and HTML structure")
        return []
    
    # Save links to CSV for reference
    links_csv_path = get_output_path("course_links", "csv")
    save_course_links_to_csv(course_links, links_csv_path)
    
    # Limit courses in test mode
    if test_mode:
        course_links = course_links[:5]
        print(f"🧪 Test mode: Scraping first {len(course_links)} courses...")
    elif max_courses:
        course_links = course_links[:max_courses]
        print(f"📚 Scraping {len(course_links)} courses (limited)...")
    else:
        print(f"📚 Scraping all {len(course_links)} courses...")
    
    # Step 2: Scrape each course page
    print("\n" + "=" * 60)
    print("Step 2: Scraping individual course pages")
    print("=" * 60)
    
    for i, (course_code, course_url, course_type) in enumerate(course_links, 1):
        print(f"\n[{i}/{len(course_links)}] Scraping {course_code}...")
        
        course = scrape_course_page(course_url, course_code, course_type)
        
        if course:
            courses.append(course)
            print(f"  ✅ {course['course_code']}: {course['course_name']}")
            if course.get('credits'):
                print(f"     Credits: {course['credits']}")
            if course.get('prerequisites'):
                print(f"     Prerequisites: {', '.join(course['prerequisites'])}")
        else:
            print(f"  ❌ Failed to scrape {course_code}")
        
        # Be respectful - robots.txt compliant delay
        if i < len(course_links):  # Don't sleep after last course
            time.sleep(2.5)
        
        # Save progress periodically (every 10 courses)
        if i % 10 == 0:
            save_courses(courses, get_output_path("progress"))
            print(f"  💾 Progress saved ({len(courses)} courses so far)")
    
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
    
    print(f"💾 Saved {len(courses)} courses to {output_path}")


def get_output_path(filename: str = "courses_raw", extension: str = "json") -> Path:
    """Get output file path.
    
    Args:
        filename: Base filename (without extension)
        extension: File extension (json, csv, etc.)
        
    Returns:
        Path object
    """
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    return output_dir / f"{filename}.{extension}"


def save_course_links_to_csv(course_links: List[Tuple[str, str, str]], output_path: Path) -> None:
    """Save course links to CSV file.
    
    Args:
        course_links: List of tuples (course_code, course_url, course_type)
        output_path: Path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['course_code', 'course_url', 'course_type'])
        # Write data
        for course_code, course_url, course_type in course_links:
            writer.writerow([course_code, course_url, course_type])
    
    print(f"💾 Saved {len(course_links)} course links to {output_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("Chalmers Course Scraper")
    print("=" * 60)
    print()
    
    # Programme URL for Data Science and AI MSc
    programme_url = "https://www.chalmers.se/en/education/find-masters-programme/data-science-and-ai-msc/"
    
    # Scrape all courses (change test_mode=True for testing with 5 courses)
    print("🚀 Starting FULL scraping mode (all courses)")
    print("   Set test_mode=True to test with only 5 courses\n")
    
    test_courses = collect_courses(
        programme_url=programme_url,
        test_mode=False  # Set to True to test with only 5 courses
    )
    
    if not test_courses:
        print("\n⚠️  No courses collected!")
        print("Please check:")
        print("1. Programme URL is correct")
        print("2. Internet connection")
        print("3. HTML structure hasn't changed")
        return
    
    # Save test results
    test_output = get_output_path("courses_test")
    save_courses(test_courses, test_output)
    
    # Inspect the results
    print("\n" + "=" * 60)
    print("Sample Course Data:")
    print("=" * 60)
    print(json.dumps(test_courses[0], indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"✅ Successfully scraped {len(test_courses)} courses")
    print(f"💾 Course data saved to: {test_output}")
    print(f"📋 Course links saved to: {get_output_path('course_links', 'csv')}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. ✅ Check the test output above")
    print("2. ✅ Verify all fields are extracted correctly")
    print("3. ✅ Inspect the saved JSON file")
    print("4. ✅ Update selectors in scrape_course_page() if needed")
    print("5. ✅ When ready, set test_mode=False to scrape all courses")
    print()
    
    # Show statistics
    compulsory = sum(1 for c in test_courses if c.get('course_type') == 'compulsory')
    elective = sum(1 for c in test_courses if c.get('course_type') == 'elective')
    print(f"📊 Course types: {compulsory} compulsory, {elective} elective")


if __name__ == "__main__":
    main()

