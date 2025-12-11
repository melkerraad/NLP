# LangChain RAG Workflow Documentation

This document explains the complete workflow of the LangChain-based RAG system for answering questions about Chalmers courses.

## Overview

The LangChain RAG pipeline consists of three main phases:
1. **Data Preparation** - Preprocessing course data (one-time)
2. **Vector Database Setup** - Creating embeddings and vector store (one-time, unless data changes)
3. **Querying** - Running queries against the RAG system (every time you want answers)

---

## Phase 1: Data Preparation (One-Time Setup)

**When to run:** Only once, or when you need to update course data

**Files involved:**
- `../project/src/preprocessing/clean_courses.py` - Cleans course data
- `data/processed/courses_clean.json` - Output: Cleaned course data

**What happens:**
1. Loads raw course data
2. Cleans and preprocesses text
3. Validates data structure
4. Saves cleaned data to `data/processed/courses_clean.json`

**How to run:**
```bash
# From parent project directory
cd ../project
python src/preprocessing/clean_courses.py
```

---

## Phase 2: Vector Database Setup (One-Time Setup)

**When to run:** After preprocessing, or when course data changes

**Files involved:**
- `src/retrieval/langchain_setup.py` - Creates embeddings and vector database
- `src/retrieval/vector_store.py` - Vector store factory (with GPU support)
- `data/processed/courses_clean.json` - Input: Cleaned course data
- `data/chroma_db/` - Output: Vector database (persistent)

**What happens:**
1. Loads cleaned courses from `data/processed/courses_clean.json`
2. Converts courses to LangChain Documents
3. Creates embeddings using SentenceTransformer (`all-MiniLM-L6-v2`)
   - **Automatically uses GPU if available** (L40s/L4)
   - Falls back to CPU if GPU not available
4. Stores embeddings in ChromaDB collection named `chalmers_courses`
5. Database is saved to `data/chroma_db/` (persists between runs)

**How to run:**

**On Minerva (SLURM batch job with GPU):**
```bash
cd project_langchain
sbatch run_setup_langchain.sh
```

**Locally:**
```bash
python -m src.retrieval.langchain_setup
```

**Expected Setup Time:**
- **With GPU (L40s/L4)**: 1-3 minutes for ~50 courses
- **Without GPU (CPU)**: 5-15 minutes for ~50 courses
- **First run**: May take longer due to model download (~90MB)

**Verify Setup:**
```bash
# Check if database was created
ls -la data/chroma_db/
# Should see chroma.sqlite3 and UUID directory
```

---

## Phase 3: RAG Querying (Every Time)

**When to run:** Every time you want to ask questions about courses

**Files involved:**
- `test_langchain_rag.py` - Main RAG test script with timing
- `test_retrieval_only.py` - Test retrieval without LLM
- `src/retrieval/vector_store.py` - Vector store (with GPU support)
- `src/generation/llm_factory.py` - LLM factory (with device_map fix)
- `src/generation/rag_chain.py` - RAG chain builder
- `src/utils/timing.py` - Timing utilities
- `data/chroma_db/` - Input: Vector database (read-only)

**What happens (for each query):**

1. **Load Vector Database**
   - Opens existing ChromaDB from `data/chroma_db/`
   - Loads embedding model (SentenceTransformer) on GPU if available
   - No need to recreate embeddings

2. **Load Language Model**
   - Downloads Llama 3.2 1B-Instruct model on first run (~2GB)
   - Caches model locally (subsequent runs are faster)
   - Uses `device_map="auto"` for optimal GPU utilization
   - Compiles model with `torch.compile` for faster inference

3. **For Each Query:**
   - **Retrieval:** 
     - Generates query embedding (GPU if available)
     - Searches vector database for top-k most relevant courses
     - **Timing:** ~6-10 ms after warmup (first query slower: ~130 ms)
   - **Context Building:** Formats retrieved courses into prompt context
   - **Generation:** 
     - Sends query + context to Llama model
     - **Timing:** ~300-400 ms after warmup (first query slower: ~1.3 seconds)
   - **Response:** Returns answer with source citations

**How to run:**

**On Minerva (SLURM batch job with GPU):**
```bash
cd project_langchain
sbatch run_langchain_rag.sh

# Monitor output
tail -f rag_output_*.log
```

**Locally:**
```bash
python test_langchain_rag.py
```

**Test Retrieval Only (without LLM):**
```bash
python test_retrieval_only.py
```

**Output:** The test script runs 5 test queries and prints:
- GPU device information
- Step-by-step timing for each component
- Retrieved courses for each query
- Generated answer with sources
- Performance metrics (retrieval time, generation time, total time)
- Overall performance summary with statistics

---

## Performance Metrics

### Expected Performance on L40S GPU

**Setup Times:**
- Config loading: < 10 ms
- Vector store loading: ~2-3 seconds
- LLM loading: ~3 seconds (cached), ~2-5 minutes (first time)
- RAG chain building: < 2 ms

**Per Query Performance:**
- **Retrieval:**
  - First query: ~130 ms (includes warmup)
  - Subsequent queries: ~6-10 ms
  - Min observed: ~6 ms
- **Generation:**
  - First query: ~1.3 seconds (includes warmup)
  - Subsequent queries: ~300-400 ms
  - Min observed: ~300 ms
- **Total per query:** ~300-500 ms after warmup

**Total Runtime:**
- First run (model download): 3-7 minutes
- Subsequent runs: 30-60 seconds for 5 queries

### Key Optimizations

- **GPU acceleration** for both embeddings and LLM
- **Model compilation** with `torch.compile` for faster inference
- **Device auto-detection** - automatically uses GPU when available
- **Model caching** - downloads only once, cached for future runs
- **Efficient retrieval** - GPU-accelerated embeddings

---

## Key Features

### GPU Auto-Detection

The system automatically detects and uses GPU when available:
- **Embeddings**: Uses GPU for SentenceTransformer (if CUDA available)
- **LLM**: Uses GPU for Llama model (if CUDA available)
- Falls back to CPU if GPU not available
- All device selection is automatic - no manual configuration needed

### Timing Utilities

All operations are automatically timed:
- Step-by-step timing for each component
- Per-query performance metrics
- Overall performance summary with statistics (min, max, average, total)
- Timing displayed in milliseconds for easy reading

### Device Management

- Embeddings automatically use GPU if available (via `device_map` parameter)
- LLM uses `device_map="auto"` for optimal GPU utilization
- Pipeline creation correctly handles accelerate device mapping
- No device conflicts or errors

---

## File Structure

### Core Source Files

| File | Purpose |
|------|---------|
| `src/retrieval/langchain_setup.py` | Converts courses to LangChain Documents and creates vector store |
| `src/retrieval/vector_store.py` | Factory for creating/loading ChromaDB vector stores (with GPU support) |
| `src/generation/llm_factory.py` | Factory for creating LLMs (with device_map fix) |
| `src/generation/prompt_templates.py` | Centralized prompt template management |
| `src/generation/rag_chain.py` | Modular RAG chain builder |
| `src/ui/gradio_app.py` | Gradio interface for interactive queries |
| `src/utils/config_loader.py` | Loads and validates configuration |
| `src/utils/timing.py` | Timing utilities for performance tracking |

### Scripts

| File | Purpose |
|------|---------|
| `test_langchain_rag.py` | Main RAG test script with detailed timing |
| `test_retrieval_only.py` | Test retrieval without LLM generation |
| `run_setup_langchain.sh` | SLURM batch script for vector database setup |
| `run_langchain_rag.sh` | SLURM batch script for RAG testing |
| `run_ui_minerva.sh` | SLURM batch script for Gradio UI |

### Data Files

| Directory/File | Purpose | When Created |
|----------------|---------|--------------|
| `data/processed/courses_clean.json` | Cleaned course data | Preprocessing phase |
| `data/chroma_db/` | Vector database with embeddings | Setup phase (persistent) |

### Configuration

| File | Purpose |
|------|---------|
| `config/config.yaml` | All settings (model, retrieval, vector store, UI) |
| `.env` | Contains `HF_TOKEN` for Hugging Face access |

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

3. **Request access to Llama model:**
   - Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - Click "Agree and access repository"

4. **Prepare course data:**
   ```bash
   cd ../project
   python src/preprocessing/clean_courses.py
   ```

5. **Create vector database:**
   ```bash
   cd ../project_langchain
   sbatch run_setup_langchain.sh
   ```

### Running Queries (Every Time)

```bash
# On Minerva with GPU
cd project_langchain
sbatch run_langchain_rag.sh

# Monitor output
tail -f rag_output_*.log
```

---

## Troubleshooting

**Vector store not found:**
- Run setup script: `sbatch run_setup_langchain.sh`
- Verify `data/chroma_db/` exists

**LLM pipeline error (device_map):**
- Should be fixed in latest version
- Update file: `git checkout origin/main -- src/generation/llm_factory.py`

**Slow performance:**
- Check GPU availability in logs: `CUDA available: True`
- Verify embeddings using GPU: `[INFO] Creating embeddings on device: cuda`
- First query is slower due to warmup - subsequent queries are faster

**No GPU detected:**
- Check SLURM script includes: `#SBATCH --gres=gpu:L40s:1`
- Verify GPU availability: `nvidia-smi` in batch job

**Model access denied:**
- Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Make sure `HF_TOKEN` is set in `.env`

---

## Next Steps

After successful setup and testing:

1. **Experiment with queries** - Modify `test_langchain_rag.py` to test your own queries
2. **Tune retrieval parameters** - Adjust `top_k` in `config/config.yaml`
3. **Try the Gradio UI** - Run `run_ui_minerva.sh` for interactive testing
4. **Optimize prompts** - Edit prompt templates in `src/generation/prompt_templates.py`
5. **Add more features** - Extend the system with new LLM providers or vector stores

