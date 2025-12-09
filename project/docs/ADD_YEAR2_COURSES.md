# How to Add Year 2 Courses

The script now supports both Year 1 and Year 2 courses. Here's how to add Year 2:

## Steps

1. **Go to the programme page:**
   https://www.chalmers.se/en/education/find-masters-programme/data-science-and-ai-msc/

2. **Scroll down to find the "Year 2" section** (or look for "Programme plan" section)

3. **Right-click on the Year 2 course list** and select "Inspect" (or press F12)

4. **Find the `<ul>` element** containing Year 2 courses (it should have similar classes like `grid gap-3 px-3 py-6 no-list-formatting`)

5. **Copy the entire `<ul>...</ul>` block** with all the `<li>` elements

6. **Open** `src/data_collection/create_links_csv.py`

7. **Paste the Year 2 HTML** into the `YEAR_2_HTML` variable (replace the empty `<ul>`)

8. **Run the script:**
   ```bash
   python src/data_collection/create_links_csv.py
   ```

9. **Check the output** - it should show courses from both years

## Example

The `YEAR_2_HTML` section should look like:

```python
YEAR_2_HTML = """
<ul class="grid gap-3 px-3 py-6 no-list-formatting">
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/XXX123/">Course name </a>(compulsory)</li>
<li><a href="/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/YYY456/">Another course </a>(elective)</li>
...
</ul>
"""
```

## Alternative: Manual Entry

If you can't find the HTML easily, you can also manually add courses to the CSV file or create a simple list and add them programmatically.

## After Adding Year 2

Once you've updated the script with Year 2 courses:
1. Run `python src/data_collection/create_links_csv.py`
2. It will create/update `data/raw/course_links.csv` with all courses from both years
3. The main scraper will automatically use this updated CSV file

