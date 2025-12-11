# RAG Pipeline Workflow Documentation

This document explains the complete workflow of the RAG (Retrieval-Augmented Generation) system for answering questions about Chalmers courses.

## Overview

The RAG pipeline consists of three main phases:
1. **Data Collection** - Scraping course data from Chalmers website (one-time)
2. **Setup** - Creating vector embeddings and database (one-time, unless data changes)
3. **Querying** - Running queries against the RAG system (every time you want answers)

---

## Phase 1: Data Collection (One-Time Setup)

**When to run:** Only once, or when you need to update course data

**Files involved:**
- `src/data_collection/chalmers_scraper.py` - Scrapes course pages from Chalmers
- `src/data_collection/create_links_csv.py` - Creates CSV of course links
- `data/raw/course_links.csv` - Input: List of course URLs to scrape
- `data/raw/courses_test.json` - Output: Raw scraped course data

**What happens:**
1. Reads course links from `data/raw/course_links.csv`
2. Scrapes each course page to extract:
   - Course code (e.g., "DAT450")
   - Course name
   - Description
   - Credits
   - Prerequisites
   - Learning outcomes
   - Other metadata
3. Saves raw JSON data to `data/raw/courses_test.json`

**How to run:**
```bash
python src/data_collection/chalmers_scraper.py
```

---

## Phase 2: Preprocessing (One-Time Setup)

**When to run:** After data collection, or when raw data changes

**Files involved:**
- `src/preprocessing/clean_courses.py` - Cleans and preprocesses course data
- `data/raw/courses_test.json` - Input: Raw scraped data
- `data/processed/courses_clean.json` - Output: Cleaned course data

**What happens:**
1. Loads raw course data from `data/raw/courses_test.json`
2. Removes empty/null fields
3. Cleans text (removes extra whitespace, normalizes formatting)
4. Validates data structure
5. Saves cleaned data to `data/processed/courses_clean.json`

**How to run:**
```bash
python src/preprocessing/clean_courses.py
```

---

## Phase 3: Vector Database Setup (One-Time Setup)

**When to run:** After preprocessing, or when course data changes

**Files involved:**
- `src/retrieval/setup_retrieval.py` - Creates embeddings and vector database
- `data/processed/courses_clean.json` - Input: Cleaned course data
- `data/chroma_db/` - Output: Vector database (persistent)

**What happens:**
1. Loads cleaned courses from `data/processed/courses_clean.json`
2. Initializes ChromaDB vector database
3. Creates embeddings for each course using SentenceTransformer (`all-MiniLM-L6-v2`)
4. Stores embeddings in ChromaDB collection named `chalmers_courses`
5. Database is saved to `data/chroma_db/` (persists between runs)

**How to run:**
```bash
# On Minerva (SLURM batch job)
sbatch run_setup_retrieval.sh

# Or locally
python src/retrieval/setup_retrieval.py
```

**Note:** This step can take a few minutes depending on the number of courses. The database persists, so you only need to rebuild if course data changes.

---

## Phase 4: RAG Querying (Every Time)

**When to run:** Every time you want to ask questions about courses

**Files involved:**
- `test_rag.py` - Main RAG test script
- `src/retrieval/setup_retrieval.py` - Contains `CourseRetriever` class
- `src/generation/llama_generator.py` - Contains `LlamaRAGGenerator` class
- `data/chroma_db/` - Input: Vector database (read-only)
- `rag_output_<job_id>.log` - Output: Query results and logs

**What happens (for each query):**

1. **Load Vector Database**
   - Opens existing ChromaDB from `data/chroma_db/`
   - Loads embedding model (SentenceTransformer)
   - No need to recreate embeddings

2. **Load Language Model**
   - Downloads Llama 3.2 1B-Instruct model on first run (~2GB)
   - Caches model locally (subsequent runs are faster)
   - Compiles model with `torch.compile` for faster inference
   - Runs warmup generation to initialize compiled model

3. **For Each Query:**
   - **Retrieval:** Searches vector database for top 3 most relevant courses
   - **Context Building:** Formats retrieved courses into prompt context
   - **Generation:** Sends query + context to Llama model
   - **Response:** Returns answer with source citations

**How to run:**
```bash
# On Minerva (SLURM batch job with GPU)
sbatch run_minerva_rag.sh

# Or locally (requires GPU for reasonable speed)
python test_rag.py
```

**Output:** The script runs 5 test queries and prints:
- Retrieved courses for each query
- Generated answer
- Source citations
- Generation time per query

---

## File Structure and Purpose

### Core Source Files

| File | Purpose |
|------|---------|
| `src/data_collection/chalmers_scraper.py` | Scrapes course data from Chalmers website |
| `src/data_collection/create_links_csv.py` | Creates CSV file with course links |
| `src/preprocessing/clean_courses.py` | Cleans and preprocesses scraped course data |
| `src/retrieval/setup_retrieval.py` | Creates vector embeddings and ChromaDB database |
| `src/generation/llama_generator.py` | Generates answers using Llama 3.2 model |

### Scripts

| File | Purpose |
|------|---------|
| `test_rag.py` | Main RAG query script - runs test queries |
| `run_setup_retrieval.sh` | SLURM batch script for setting up vector database |
| `run_minerva_rag.sh` | SLURM batch script for running RAG queries on GPU |

### Data Files

| Directory/File | Purpose | When Created |
|----------------|---------|---------------|
| `data/raw/course_links.csv` | List of course URLs to scrape | Manual/one-time |
| `data/raw/courses_test.json` | Raw scraped course data | Phase 1 |
| `data/processed/courses_clean.json` | Cleaned course data | Phase 2 |
| `data/chroma_db/` | Vector database with embeddings | Phase 3 (persistent) |

### Configuration

| File | Purpose |
|------|---------|
| `.env` | Contains `HF_TOKEN` for Hugging Face access |
| `requirements.txt` | Python dependencies |
| `MINERVA_SETUP.md` | Setup instructions for Minerva HPC |

---

## Typical Workflow Summary

### First-Time Setup (Do Once)

1. **Install dependencies:**
   ```bash
   python3 -m venv venv_minerva
   source venv_minerva/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up Hugging Face token:**
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```

3. **Collect course data:**
   ```bash
   python src/data_collection/chalmers_scraper.py
   ```

4. **Preprocess data:**
   ```bash
   python src/preprocessing/clean_courses.py
   ```

5. **Create vector database:**
   ```bash
   sbatch run_setup_retrieval.sh  # On Minerva
   # or
   python src/retrieval/setup_retrieval.py  # Locally
   ```

### Running Queries (Every Time)

```bash
# On Minerva with GPU
sbatch run_minerva_rag.sh

# Check output
tail -f rag_output_<job_id>.log

# Or locally (if you have GPU)
python test_rag.py
```

---

## Performance Notes

- **Vector Database Setup:** ~5-10 minutes (one-time, depends on number of courses)
- **Model Loading:** ~1-2 minutes first time (downloads ~2GB), ~10-30 seconds cached
- **Per Query:** ~0.5-2 seconds on GPU (L40s), ~10-30 seconds on CPU
- **Total for 5 queries:** ~1-3 minutes on GPU

The optimizations in place:
- `max_new_tokens=200` (reduced from 500)
- Greedy decoding (temperature=0.1) for faster generation
- `torch.compile` for model optimization
- KV cache enabled
- Model warmup to initialize compiled model

---

## Troubleshooting

**Database not found:**
- Run Phase 3 (setup_retrieval.py) first

**Model access denied:**
- Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Make sure `HF_TOKEN` is set in `.env`

**Slow generation:**
- Make sure you're using GPU (check `nvidia-smi` on Minerva)
- Verify model is cached (second run should be faster)
- Check generation time in output logs

**No courses retrieved:**
- Verify vector database exists and has data
- Check `data/chroma_db/` directory
- Re-run setup_retrieval.py if needed

