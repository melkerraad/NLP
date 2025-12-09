# Web Scraping Guidelines for Chalmers

## ✅ Robots.txt Analysis

**Status: ✅ Scraping is ALLOWED**

Chalmers robots.txt (https://www.chalmers.se/robots.txt):
```
User-agent: *
Allow: /
Disallow: /sok/
Disallow: /en/search/
```

**What this means:**
- ✅ **Allowed**: Course pages, general content
- ❌ **Disallowed**: Search endpoints (`/sok/`, `/en/search/`)
- ✅ **Course catalog pages**: Allowed to scrape

## Best Practices

### 1. Rate Limiting
- **Delay between requests**: 2-3 seconds minimum
- **Don't hammer the server**: Spread requests over time
- **Use exponential backoff**: If you get rate-limited, back off

### 2. User-Agent
Always identify yourself:
```python
headers = {
    'User-Agent': 'Mozilla/5.0 ... (Educational Research Project - DAT450 NLP Course)'
}
```

### 3. Error Handling
- Handle 429 (Too Many Requests) gracefully
- Retry with exponential backoff
- Log errors for review
- Don't crash on single page failures

### 4. Respectful Scraping
- ✅ Scrape during off-peak hours if possible
- ✅ Cache results (don't re-scrape unnecessarily)
- ✅ Use session objects to reuse connections
- ✅ Set reasonable timeouts (10 seconds)
- ❌ Don't use multiple threads/processes aggressively
- ❌ Don't scrape search endpoints (disallowed)

## Legal & Ethical Considerations

### Educational Use
- This is for **educational/research purposes** (DAT450 course project)
- **Non-commercial** use
- **Respectful** scraping practices

### Terms of Service
- Check Chalmers Terms of Service if available
- Generally, public course information is fine to scrape
- Don't scrape personal/private information

### If You're Unsure
- Contact Chalmers IT/webmaster if you have concerns
- Explain it's for an educational NLP project
- They're usually supportive of educational research

## Implementation Checklist

- [x] Robots.txt allows scraping
- [ ] User-Agent identifies as educational project
- [ ] 2-3 second delays between requests
- [ ] Error handling for rate limits
- [ ] Timeout set (10 seconds)
- [ ] Avoid search endpoints
- [ ] Save progress periodically
- [ ] Log errors for debugging

## Example: Respectful Scraper

```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Respectful headers
headers = {
    'User-Agent': 'Educational Research Project - DAT450 NLP Course',
}

# Scrape with delays
for url in course_urls:
    try:
        response = session.get(url, headers=headers, timeout=10)
        # Process response...
        time.sleep(2.5)  # Respectful delay
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        time.sleep(5)  # Longer delay on error
```

## Monitoring

Watch for these signs you're being too aggressive:
- 429 (Too Many Requests) responses
- Connection timeouts
- IP blocking
- Slower response times

If you see these, **slow down** and increase delays.

---

**Remember**: Be a good internet citizen! 🚀

