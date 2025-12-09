# RAG Pipeline - Chalmers Course Chatbot

A simple Retrieval-Augmented Generation (RAG) system that answers questions about Chalmers courses using Llama 3.2 1B (~2GB model).

## Quick Start

### Prerequisites

- Python 3.9+
- Hugging Face token with access to Llama 3.2
  - Get token: https://huggingface.co/settings/tokens
  - Request access: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (1B model, ~2GB)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your Hugging Face token
# Copy .env.example to .env and add your token:
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows

# Then edit .env and replace 'your_token_here' with your actual token
```

Alternatively, you can set the environment variable directly:
```bash
export HF_TOKEN="your_token_here"  # Linux/Mac
$env:HF_TOKEN="your_token_here"    # Windows PowerShell
```

### Run Proof of Concept

```bash
python test_rag.py
```

This will:
1. Load the vector database (already created from scraped courses)
2. Retrieve relevant courses for a test query
3. Generate a response using Llama 3.2 1B (~2GB download on first run)

## Project Structure

```
project/
├── data/
│   └── chroma_db/          # Vector database (already created)
├── src/
│   ├── retrieval/          # CourseRetriever class
│   └── generation/         # LlamaRAGGenerator class
├── test_rag.py             # Proof of concept test
└── requirements.txt        # Dependencies
```

## Usage

The `test_rag.py` script demonstrates the complete RAG pipeline. Modify the query in the script to test different questions about Chalmers courses.
