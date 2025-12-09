"""Create course_links.csv from manually provided course links.

Since the page loads links dynamically, you can use this script to create
the CSV file from the links you can see on the page.
"""

import csv
from pathlib import Path
import re

# Paste the HTML with course links here
# You can paste Year 1 and Year 2 courses separately or together

# YEAR 1 COURSES
YEAR_1_HTML = """
<ul class="grid gap-3 px-3 py-6 no-list-formatting">
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT695/">Introduction to data science </a>(compulsory)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA947/">Nonlinear optimisation </a>(compulsory)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT246/">Empirical software engineering </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT441/">Advanced topics in machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT465/">Causality and machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT625/">Structured machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/EEN100/">Statistics and machine learning in high dimensions </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/FFR105/">Stochastic optimization algorithms </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/FFR135/">Artificial neural networks </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE188/">Computational methods for Bayesian statistics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/RRY025/">Image processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY340/">Deep machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TIN093/">Algorithms </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA265/">Numerical linear algebra </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA882/">High performance computing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE550/">Stochastic processes and Bayesian inference </a>(compulsory)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT450/">Machine learning for natural language processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT570/">Continuous optimization in data science </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/EEN020/">Computer vision </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE095/">Options and mathematics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE172/">Basic stochastic processes and financial applications </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE190/">Statistical learning with regression models </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY130/">Applied signal processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY316/">Advanced probabilistic machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA251/">Algorithms, advanced course </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA357/">Databases </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA507/">Computational methods in bioinformatics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA596/">Distributed systems </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TEK656/">Creating technology-based ventures </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA522/">Large scale optimization </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT410/">Design of AI systems </a>(compulsory)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT341/">Applied machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT675/">Artificial intelligence for molecules </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/EEN210/">Applied digital health </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA233/">Algorithms for machine learning and inference </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TIF150/">Information theory for complex systems </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT471/">Computational techniques for large-scale data </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT475/">Advanced databases </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT530/">Research-oriented course in data science and AI </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE166/">Linear and integer optimization with applications </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE441/">Statistical learning for big data </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY098/">Image analysis </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMS016/">Spatial statistics and image analysis </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMS088/">Financial time series </a>(elective)</li>
</ul>
"""

# YEAR 2 COURSES
YEAR_2_HTML = """
<ul class="grid gap-3 px-3 py-6 no-list-formatting">
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT246/">Empirical software engineering </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT441/">Advanced topics in machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT465/">Causality and machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT625/">Structured machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/EEN100/">Statistics and machine learning in high dimensions </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/FFR105/">Stochastic optimization algorithms </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/FFR135/">Artificial neural networks </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE188/">Computational methods for Bayesian statistics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/RRY025/">Image processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY340/">Deep machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TIN093/">Algorithms </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA265/">Numerical linear algebra </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA882/">High performance computing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DATX60/">Master's thesis in Computer science and engineering </a>(diploma thesis)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVEX60/">Master's thesis in Mathematics </a>(diploma thesis)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT450/">Machine learning for natural language processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT570/">Continuous optimization in data science </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/EEN020/">Computer vision </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE095/">Options and mathematics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE172/">Basic stochastic processes and financial applications </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVE190/">Statistical learning with regression models </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY130/">Applied signal processing </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/SSY316/">Advanced probabilistic machine learning </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA251/">Algorithms, advanced course </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA357/">Databases </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA507/">Computational methods in bioinformatics </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TDA596/">Distributed systems </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TEK656/">Creating technology-based ventures </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/TMA522/">Large scale optimization </a>(elective)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DATX05/">Master's thesis in Computer science and engineering </a>(diploma thesis)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/MVEX03/">Master's thesis in Mathematics </a>(diploma thesis)</li>
</ul>
"""

def parse_links_from_html(html_text):
    """Parse course links from HTML."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_text, 'html.parser')
    course_links = []
    
    # Find all <li> elements with course links
    list_items = soup.find_all('li')
    
    for item in list_items:
        link = item.find('a', href=re.compile(r'/course-syllabus/'))
        if link:
            href = link.get('href', '')
            course_code = href.split('/course-syllabus/')[-1].rstrip('/')
            
            # Build full URL
            if href.startswith('/'):
                full_url = f"https://www.chalmers.se{href}"
            else:
                full_url = href
            
            # Determine course type
            item_text = item.get_text().lower()
            if "compulsory" in item_text:
                course_type = "compulsory"
            elif "diploma thesis" in item_text or "thesis" in item_text:
                course_type = "thesis"
            else:
                course_type = "elective"
            
            course_links.append((course_code, full_url, course_type))
    
    return course_links


def main():
    """Create CSV from provided links."""
    print("Parsing course links from HTML...")
    
    # Parse Year 1 courses
    year1_links = parse_links_from_html(YEAR_1_HTML)
    print(f"Year 1: Found {len(year1_links)} courses")
    
    # Parse Year 2 courses
    year2_links = parse_links_from_html(YEAR_2_HTML)
    print(f"Year 2: Found {len(year2_links)} courses")
    
    # Combine all courses (avoid duplicates)
    all_course_links = []
    seen_codes = set()
    
    for course_code, course_url, course_type in year1_links + year2_links:
        if course_code not in seen_codes:
            seen_codes.add(course_code)
            all_course_links.append((course_code, course_url, course_type))
    
    course_links = all_course_links
    print(f"\nTotal unique courses: {len(course_links)}")
    
    # Save to CSV
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "course_links.csv"
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['course_code', 'course_url', 'course_type'])
        for course_code, course_url, course_type in course_links:
            writer.writerow([course_code, course_url, course_type])
    
    print(f"\n✅ Saved {len(course_links)} course links to {output_path}")
    
    # Show summary
    compulsory = sum(1 for _, _, ct in course_links if ct == 'compulsory')
    elective = sum(1 for _, _, ct in course_links if ct == 'elective')
    thesis = sum(1 for _, _, ct in course_links if ct == 'thesis')
    print(f"\nSummary:")
    print(f"   - {compulsory} compulsory courses")
    print(f"   - {elective} elective courses")
    if thesis > 0:
        print(f"   - {thesis} thesis courses")
    print(f"   - {len(year1_links)} courses from Year 1")
    print(f"   - {len(year2_links)} courses from Year 2")
    print(f"   - {len(course_links)} unique courses total")


if __name__ == "__main__":
    main()

