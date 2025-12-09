# Next Steps - RAG Course Chatbot Project

## ✅ Phase 1: Data Collection - COMPLETED

- [x] Course links collected (48 courses from Year 1 & 2)
- [x] CSV file created with all course links
- [x] Test scraping completed (5 courses)

## 🎯 Current Step: Scrape All Courses

### Step 1: Scrape All 48 Courses

Run the scraper to get full course data:

```bash
python src/data_collection/chalmers_scraper.py
```

**Expected output:**
- Scrapes all 48 courses from the CSV file
- Saves to `data/raw/courses_raw.json`
- Takes ~2-3 minutes (with 2.5s delay between requests)

**What you'll get:**
- Course codes, names, descriptions
- Credits, prerequisites
- Course type (compulsory/elective/thesis)
- URLs and metadata

---

## 📋 Phase 2: Retrieval System Setup (Next)

After scraping all courses, you'll move to Phase 2:

### Tasks:
1. **Set up Vector Database**
   - Choose: ChromaDB (easiest) or FAISS
   - Install: `pip install chromadb`

2. **Select Embedding Model**
   - Recommended: `sentence-transformers` with `all-MiniLM-L6-v2`
   - Or: OpenAI embeddings (requires API key)

3. **Preprocess Course Data**
   - Clean and structure the scraped data
   - Chunk long descriptions (if needed)
   - Prepare for embedding

4. **Create Embeddings**
   - Generate embeddings for all course documents
   - Store in vector database

5. **Test Retrieval**
   - Test with sample queries
   - Evaluate retrieval quality

### Files to Create:
- `src/preprocessing/clean_courses.py` - Data cleaning
- `src/retrieval/setup_vector_db.py` - Vector DB setup
- `src/retrieval/embed_courses.py` - Embedding pipeline
- `src/retrieval/retrieve.py` - Retrieval system

---

## 🚀 Quick Start Commands

### 1. Scrape All Courses (NOW)
```bash
cd project
python src/data_collection/chalmers_scraper.py
```

### 2. After Scraping - Check Results
```bash
# View the scraped data
cat data/raw/courses_raw.json | head -50
```

### 3. Next: Set Up Retrieval System
```bash
# Install vector database
pip install chromadb sentence-transformers

# Then create the retrieval system (we'll help with this next)
```

---

## 📊 Progress Checklist

### Phase 1: Data Collection ✅
- [x] Research course data sources
- [x] Collect course links (48 courses)
- [x] Create CSV with links
- [ ] **Scrape all courses** ← YOU ARE HERE
- [ ] Clean and validate data

### Phase 2: Retrieval System (Next)
- [ ] Set up vector database
- [ ] Select embedding model
- [ ] Preprocess course data
- [ ] Generate embeddings
- [ ] Test retrieval

### Phase 3: Generation System
- [ ] Set up LLM API
- [ ] Design prompt templates
- [ ] Implement generation pipeline

### Phase 4: Integration & Frontend
- [ ] Integrate retrieval + generation
- [ ] Build user interface
- [ ] End-to-end testing

### Phase 5: Evaluation
- [ ] Create test dataset
- [ ] Run evaluations
- [ ] Refine system

---

## 💡 Tips

1. **After scraping:** Check `data/raw/courses_raw.json` to verify data quality
2. **Data cleaning:** You may need to fix extraction issues (credits, names, etc.)
3. **Start simple:** Use ChromaDB and sentence-transformers for quick setup
4. **Test incrementally:** Test retrieval with a few courses first, then scale up

---

**Ready to scrape all courses? Run the command above!** 🚀

