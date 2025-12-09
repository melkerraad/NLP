"""Debug script to inspect the programme page HTML structure."""

import requests
from bs4 import BeautifulSoup
from pathlib import Path

def debug_programme_page():
    """Download and inspect the programme page."""
    url = "https://www.chalmers.se/en/education/find-masters-programme/data-science-and-ai-msc/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Fetching: {url}")
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Save HTML for inspection
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_file = output_dir / "programme_page_debug.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())
    print(f"Saved HTML to: {html_file}")
    
    # Find all links
    all_links = soup.find_all('a', href=True)
    print(f"\nFound {len(all_links)} total links")
    
    # Find course syllabus links
    course_links = [link for link in all_links if 'course-syllabus' in link.get('href', '').lower()]
    print(f"Found {len(course_links)} course-syllabus links")
    
    if course_links:
        print("\nFirst 5 course links:")
        for i, link in enumerate(course_links[:5], 1):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            print(f"{i}. {text}")
            print(f"   URL: {href}")
            print()
    
    # Find all <ul> elements
    ul_elements = soup.find_all('ul')
    print(f"\nFound {len(ul_elements)} <ul> elements")
    
    for i, ul in enumerate(ul_elements[:3], 1):  # Show first 3
        classes = ul.get('class', [])
        li_count = len(ul.find_all('li'))
        print(f"\nUL {i}:")
        print(f"  Classes: {classes}")
        print(f"  List items: {li_count}")
        
        # Check if it has course links
        course_links_in_ul = ul.find_all('a', href=lambda x: x and 'course-syllabus' in x.lower())
        if course_links_in_ul:
            print(f"  ✅ Contains {len(course_links_in_ul)} course syllabus links!")
            print(f"  First link: {course_links_in_ul[0].get('href')}")

if __name__ == "__main__":
    debug_programme_page()

