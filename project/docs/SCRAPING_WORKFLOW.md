# Scraping Workflow Guide

## Two-Step Process

The scraper works in two steps:

### Step 1: Extract Course Links from Programme Page
- Scrapes the master's programme page (e.g., Data Science and AI MSc)
- Extracts all course links from the course list
- Identifies which courses are compulsory vs elective
- Gets course codes and URLs

### Step 2: Scrape Individual Course Pages
- For each course URL found in Step 1
- Scrapes the course syllabus page
- Extracts: course code, name, description, credits, prerequisites, etc.
- Saves all data to JSON

## Usage

### Test Mode (Recommended First)
```bash
cd project
python src/data_collection/chalmers_scraper.py
```

This will:
- Scrape the programme page to get all course links
- Scrape only the **first 5 courses** (for testing)
- Save results to `data/raw/courses_test.json`

### Full Scraping Mode
Edit `chalmers_scraper.py` and change:
```python
test_courses = collect_courses(
    programme_url=programme_url,
    test_mode=False  # Change to False
)
```

This will scrape **all courses** from the programme.

## Programme URL

Default: Data Science and AI MSc
```
https://www.chalmers.se/en/education/find-masters-programme/data-science-and-ai-msc/
```

You can change this to scrape other programmes:
- Computer Science MSc
- Other master's programmes
- Or provide a different URL

## Output Format

Each course is saved as:
```json
{
  "course_code": "DAT450",
  "course_name": "Machine learning for natural language processing",
  "description": "Course description text...",
  "credits": 7.5,
  "prerequisites": ["DAT250", "DAT260"],
  "learning_outcomes": [],
  "schedule": null,
  "instructor": null,
  "department": null,
  "level": null,
  "course_type": "elective",
  "url": "https://www.chalmers.se/.../course-syllabus/DAT450/",
  "scraped_date": "2025-12-09"
}
```

## Progress Saving

The scraper automatically saves progress:
- Every 10 courses → saves to `data/raw/progress.json`
- Final results → saves to `data/raw/courses_test.json` (or `courses_raw.json`)

If the scraper crashes, you can resume from the progress file.

## Troubleshooting

### No courses found
- Check the programme URL is correct
- Verify the HTML structure hasn't changed
- Check internet connection

### Missing fields
- The scraper uses pattern matching - some fields might need manual extraction
- Inspect the HTML structure of a course page
- Update selectors in `scrape_course_page()` function

### Rate limiting
- If you get 429 errors, increase the delay (currently 2.5 seconds)
- Change `time.sleep(2.5)` to `time.sleep(5)` for slower scraping

## Next Steps After Scraping

1. **Inspect the data**: Check `data/raw/courses_test.json`
2. **Clean the data**: Remove HTML, normalize text (see preprocessing module)
3. **Validate**: Check for missing fields, duplicates
4. **Move to Phase 2**: Set up vector database and embeddings

