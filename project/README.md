# RAG Pipeline - Chalmers Course Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions about Chalmers courses using Llama 3.2 1B (~2GB model).

## 📖 Documentation

- **[RAG_WORKFLOW.md](RAG_WORKFLOW.md)** - Complete workflow documentation (what happens first, what runs every time, file purposes)
- **[MINERVA_SETUP.md](MINERVA_SETUP.md)** - Setup instructions for Minerva HPC cluster

## Quick Start

### Prerequisites

- Python 3.9+
- Hugging Face token with access to Llama 3.2
  - Get token: https://huggingface.co/settings/tokens
  - Request access: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Installation

```bash
# Create virtual environment
python3 -m venv venv_minerva
source venv_minerva/bin/activate  # On Windows: venv_minerva\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token
echo "HF_TOKEN=your_token_here" > .env
```

### Running RAG Queries

**On Minerva (with GPU):**
```bash
sbatch run_minerva_rag.sh
tail -f rag_output_<job_id>.log
```

**Locally:**
```bash
python test_rag.py
```

## Project Structure

```
project/
├── data/
│   ├── raw/                # Raw scraped course data
│   ├── processed/          # Cleaned course data
│   └── chroma_db/          # Vector database (persistent)
├── src/
│   ├── data_collection/    # Web scraping scripts
│   ├── preprocessing/      # Data cleaning
│   ├── retrieval/          # Vector database & retrieval
│   └── generation/         # Llama model generation
├── test_rag.py             # Main RAG query script
├── run_minerva_rag.sh      # SLURM batch script for queries
├── run_setup_retrieval.sh  # SLURM batch script for DB setup
└── RAG_WORKFLOW.md         # Complete workflow documentation
```

## Workflow Overview

1. **One-Time Setup:**
   - Scrape course data (`chalmers_scraper.py`)
   - Clean data (`clean_courses.py`)
   - Create vector database (`setup_retrieval.py`)

2. **Every Query:**
   - Load vector database
   - Retrieve relevant courses
   - Generate answer with Llama model

See [RAG_WORKFLOW.md](RAG_WORKFLOW.md) for detailed documentation.
