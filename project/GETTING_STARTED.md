# Getting Started - Action Plan

## ✅ Step 1: Initial Setup (Do This First - 1-2 hours)

Before scraping, get your environment ready:

1. **Set up Python environment**
   ```bash
   cd project
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Create `.env` file** (for API keys later)
   ```bash
   # In project root
   touch .env
   # Add: OPENAI_API_KEY=your_key_here (when you get one)
   ```

3. **Test your setup**
   - Try running the example scripts to make sure everything works
   - Check that you can import required libraries

---

## ✅ Step 2: Research Chalmers Course Data Sources (1-2 hours)

**Before writing any scraping code, investigate:**

1. **Find where course information is published:**
   - Chalmers course catalog website
   - Student portal (if accessible)
   - Department websites
   - Course syllabi repositories
   - API endpoints (if available - check for public APIs)

2. **Identify what information is available:**
   - Course codes and names
   - Descriptions
   - Prerequisites
   - Credits
   - Schedule/term
   - Instructors
   - Learning outcomes
   - Department/program

3. **Check access requirements:**
   - Do you need login credentials?
   - Are there rate limits?
   - Is scraping allowed? (Check robots.txt, terms of service)
   - Are there official APIs you should use instead?

4. **Document your findings** in `docs/DATA_SOURCES.md`

---

## ✅ Step 3: Design Your Data Schema (30 minutes)

Based on what you found, finalize the course document structure:

```json
{
  "course_code": "DAT450",
  "course_name": "Natural Language Processing",
  "credits": 7.5,
  "description": "...",
  "prerequisites": ["DAT250", "DAT260"],
  "learning_outcomes": ["..."],
  "schedule": "Autumn 2025",
  "instructor": "Dr. X",
  "department": "Computer Science",
  "level": "Master's",
  "url": "https://...",  // Source URL
  "scraped_date": "2025-12-09"  // When you collected it
}
```

**Save this schema** - you'll need it for validation later.

---

## ✅ Step 4: Start with a Small Test (1-2 hours)

**Don't scrape everything at once!** Start small:

1. **Pick 5-10 courses** to test with
2. **Write a simple scraper** for those courses
3. **Test your data extraction** - does it get all fields correctly?
4. **Save to JSON** and inspect the output
5. **Fix any issues** before scaling up

**Example approach:**
```python
# Start with manual URLs for a few courses
test_courses = [
    "https://chalmers.se/courses/DAT450",
    "https://chalmers.se/courses/DAT250",
    # ... 3-5 more
]

# Scrape these first, validate output, then scale
```

---

## ✅ Step 5: Build Full Scraper (2-4 hours)

Once your test scraper works:

1. **Find all course URLs** (if scraping from a listing page)
2. **Add error handling** (some pages might fail)
3. **Add rate limiting** (be respectful - don't hammer the server)
4. **Add progress tracking** (save periodically, don't lose work)
5. **Handle edge cases** (missing fields, different page formats)

**Tips:**
- Use `time.sleep()` between requests (1-2 seconds)
- Save progress every 10-20 courses (in case of crashes)
- Log errors to a file for review
- Use `requests` with proper headers (user-agent)

---

## ✅ Step 6: Data Cleaning & Validation (1-2 hours)

After scraping:

1. **Clean the data:**
   - Remove HTML tags
   - Normalize whitespace
   - Handle missing values
   - Standardize formats (dates, credits, etc.)

2. **Validate:**
   - Check all required fields are present
   - Verify data types (credits are numbers, etc.)
   - Look for duplicates
   - Check for obvious errors

3. **Save cleaned data** to `data/processed/courses_clean.json`

---

## ✅ Step 7: Document Everything (30 minutes)

Document:
- Where you got the data from
- When you scraped it
- Any issues encountered
- Data quality notes
- How to re-run the scraper

---

## 🎯 Recommended Starting Point

**If you want to start RIGHT NOW:**

1. **Quick setup** (15 min): Install dependencies, create `.env`
2. **Research** (30 min): Find Chalmers course catalog URL
3. **Test scrape** (1 hour): Scrape 5 courses manually, save to JSON
4. **Validate** (30 min): Check the output looks good
5. **Scale up** (2-3 hours): Build full scraper

**Total: ~4-5 hours for a working data collection system**

---

## 📝 Example: Quick Start Script

Here's a minimal example to get you started:

```python
# src/data_collection/chalmers_scraper.py
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import time

def scrape_course(course_url):
    """Scrape a single course page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Educational Research Project)'
    }
    
    response = requests.get(course_url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # TODO: Extract course information based on actual page structure
    course = {
        "course_code": "DAT450",  # Extract from page
        "course_name": "NLP",     # Extract from page
        "description": "...",     # Extract from page
        # ... etc
    }
    
    return course

def main():
    # Start with a few test URLs
    test_urls = [
        "https://www.chalmers.se/en/education/courses/DAT450",
        # Add more test URLs
    ]
    
    courses = []
    for url in test_urls:
        try:
            course = scrape_course(url)
            courses.append(course)
            print(f"Scraped: {course['course_code']}")
            time.sleep(2)  # Be respectful
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "courses_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(courses)} courses to {output_path}")

if __name__ == "__main__":
    main()
```

---

## ⚠️ Important Notes

1. **Check robots.txt**: `https://www.chalmers.se/robots.txt`
2. **Be respectful**: Add delays, don't overload servers
3. **Check for APIs**: Chalmers might have official APIs - use those if available!
4. **Legal/ethical**: Make sure scraping is allowed for educational purposes
5. **Alternative**: If scraping is difficult, consider:
   - Using publicly available course catalogs
   - Manual data entry for a subset (50-100 courses)
   - Using existing datasets if available

---

## 🚀 After Data Collection

Once you have your course data:
1. Move to **Phase 2**: Set up vector database and embeddings
2. But first, make sure you have at least 50-100 courses collected
3. Quality > Quantity - better to have 100 good courses than 1000 messy ones

