# LangChain RAG Setup on Minerva

This guide covers setting up and running the LangChain-based RAG system on Minerva HPC.

## Quick Setup

### 1. Clone/Pull Repository

```bash
cd ~/DAT450_project/NLP
git pull origin main
cd project_langchain
```

### 2. Create Virtual Environment (REQUIRED)

```bash
# Create virtual environment
python3 -m venv venv_minerva

# Activate it
source venv_minerva/bin/activate

# Verify you're using the venv Python
which python  # Should show: .../venv_minerva/bin/python
```

### 3. Install Dependencies

```bash
# Make sure venv is activated (you should see (venv_minerva) in your prompt)
pip install -r requirements.txt
```

**Note:** If you get "externally-managed-environment" error, make sure:
- You activated the virtual environment (`source venv_minerva/bin/activate`)
- You're using `pip` from the venv (check with `which pip`)

### 4. Set Up Hugging Face Token

```bash
# Create .env file in project_langchain directory
echo "HF_TOKEN=your_actual_token_here" > .env

# Or use nano/vim to edit:
nano .env
```

Add your token:
```
HF_TOKEN=hf_your_actual_token_here
```

### 5. Request Access to Llama 3.2 1B-Instruct

- Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Click "Agree and access repository"
- Wait for approval (usually instant)

### 6. Prepare Course Data

Make sure you have the cleaned course data:
```bash
# Check if processed courses exist
ls -la data/processed/courses_clean.json

# If not, you may need to run preprocessing first
# (from the parent project directory)
cd ../project
python src/preprocessing/clean_courses.py
```

---

## Phase 1: Vector Database Setup (One-Time)

**When to run:** After preprocessing, or when course data changes

### Using SLURM Batch Job (Recommended)

```bash
# Make sure you're in project_langchain directory
cd ~/DAT450_project/NLP/project_langchain

# Make script executable
chmod +x run_setup_langchain.sh

# Submit the job
sbatch run_setup_langchain.sh
```

**Important:** The setup script automatically uses GPU if available. To ensure GPU usage, make sure `run_setup_langchain.sh` includes:
```bash
#SBATCH --gres=gpu:L40s:1
```

### Monitor Setup Progress

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f setup_db_output_*.log

# Check for errors
tail -f setup_db_error_*.log
```

### Expected Setup Time

- **With GPU (L40s/L4)**: 1-3 minutes for ~50 courses
- **Without GPU (CPU)**: 5-15 minutes for ~50 courses
- **First run**: May take longer due to model download (~90MB SentenceTransformer model)

### Verify Setup Success

```bash
# Check if database was created
ls -la data/chroma_db/

# Should see:
# - chroma.sqlite3 file
# - UUID-named directory
```

---

## Phase 2: Running RAG Tests

**When to run:** Every time you want to test the RAG system

### Using SLURM Batch Job (Recommended)

```bash
# Make sure you're in project_langchain directory
cd ~/DAT450_project/NLP/project_langchain

# Submit the RAG test job
sbatch run_langchain_rag.sh

# Monitor output
tail -f rag_output_*.log
```

### Expected Runtime

- **First run (model download)**: 3-7 minutes
  - Model download: 2-5 minutes (~2GB Llama-3.2-1B)
  - Setup + queries: 1-2 minutes
- **Subsequent runs (model cached)**: 30-60 seconds
  - LLM loading: 10-30 seconds
  - 5 test queries: 1-3 seconds total

### Performance Expectations (L40S GPU)

**Per Query Performance:**
- **Retrieval**: 
  - Average: ~130 ms (first query slower due to warmup)
  - After warmup: ~6-10 ms per query
- **Generation**:
  - Average: ~1.3 seconds (first query slower due to warmup)
  - After warmup: ~300-400 ms per query
- **Total per query**: ~300-500 ms after warmup

**Setup Times:**
- Config loading: < 10 ms
- Vector store loading: ~2-3 seconds
- LLM loading: ~3 seconds (cached)
- RAG chain building: < 2 ms

---

## Key Features

### GPU Auto-Detection

The system automatically detects and uses GPU when available:
- **Embeddings**: Uses GPU for SentenceTransformer (if CUDA available)
- **LLM**: Uses GPU for Llama model (if CUDA available)
- Falls back to CPU if GPU not available

### Timing Utilities

All operations are timed automatically:
- Step-by-step timing for each component
- Per-query performance metrics
- Overall performance summary with statistics

### Device Management

- Embeddings automatically use GPU if available
- LLM uses `device_map="auto"` for optimal GPU utilization
- Pipeline creation handles accelerate device mapping correctly

---

## Running the Gradio UI

### Using SLURM Batch Job

```bash
# Submit UI job
sbatch run_ui_minerva.sh

# Monitor output
tail -f ui_output_*.log
```

### Port Forwarding

After the job starts, you'll see instructions for port forwarding. On your local machine:

```bash
# Replace <compute-node> with the actual compute node name from the log
ssh -L 7860:<compute-node>:7860 minerva
```

Then open http://localhost:7860 in your browser.

---

## Troubleshooting

### Vector Store Not Found

**Error:** `[ERROR] Vector store not found at: .../data/chroma_db`

**Solution:**
```bash
# Run the setup script first
sbatch run_setup_langchain.sh
```

### LLM Pipeline Error

**Error:** `The model has been loaded with accelerate and therefore cannot be moved to a specific device`

**Solution:** This should be fixed in the latest version. Make sure you have the updated `llm_factory.py`:
```bash
git checkout origin/main -- src/generation/llm_factory.py
```

### Slow Performance

**Check GPU availability:**
```bash
# In your batch job output, look for:
# CUDA available: True
# CUDA device: NVIDIA L40S (or similar)
```

**Check if embeddings are using GPU:**
```bash
# Look for in logs:
# [INFO] Creating embeddings on device: cuda
```

### Job Cancelled

**To cancel a job:**
```bash
# Find job ID
squeue -u $USER

# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Pulling Specific Files from Git

**To update only one file without affecting database:**
```bash
# Fetch latest changes
git fetch origin

# Checkout specific file
git checkout origin/main -- src/generation/llm_factory.py

# Verify no differences
git diff origin/main -- src/generation/llm_factory.py
```

**Note:** Your database (`data/chroma_db/`) is safe because it's not tracked by git. Only tracked files are affected by git operations.

---

## File Structure

### Important Files

| File | Purpose |
|------|---------|
| `run_setup_langchain.sh` | SLURM script for vector database setup |
| `run_langchain_rag.sh` | SLURM script for RAG testing |
| `run_ui_minerva.sh` | SLURM script for Gradio UI |
| `test_langchain_rag.py` | Main RAG test script with timing |
| `test_retrieval_only.py` | Test retrieval without LLM |
| `src/retrieval/langchain_setup.py` | Vector database setup script |
| `src/retrieval/vector_store.py` | Vector store factory (with GPU support) |
| `src/generation/llm_factory.py` | LLM factory (with device_map fix) |
| `src/utils/timing.py` | Timing utilities |
| `config/config.yaml` | Configuration file |

### Data Files

| Directory/File | Purpose | When Created |
|----------------|---------|--------------|
| `data/processed/courses_clean.json` | Cleaned course data | Preprocessing phase |
| `data/chroma_db/` | Vector database | Setup phase (persistent) |

---

## Common Commands

### Activate Virtual Environment

```bash
source venv_minerva/bin/activate
```

### Check GPU Availability

```bash
nvidia-smi
```

### Monitor Job Output

```bash
# Watch latest output file
tail -f rag_output_*.log

# Or specific job
tail -f rag_output_<job_id>.log
```

### Check Job Status

```bash
squeue -u $USER
```

---

## Performance Notes

- **First query is slower** due to CUDA initialization and model warmup
- **Subsequent queries are faster** (retrieval: ~6-10 ms, generation: ~300-400 ms)
- **Model caching**: First run downloads ~2GB, cached for future runs
- **GPU acceleration**: Significant speedup for both embeddings and generation

---

## Next Steps

After successful setup and testing:

1. **Experiment with different queries** - Modify `test_langchain_rag.py` to test your own queries
2. **Tune retrieval parameters** - Adjust `top_k` in `config/config.yaml`
3. **Try the Gradio UI** - Run `run_ui_minerva.sh` for interactive testing
4. **Optimize prompts** - Edit prompt templates in `src/generation/prompt_templates.py`

