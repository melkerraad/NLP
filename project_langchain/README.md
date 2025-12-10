# LangChain RAG Pipeline - Modular & Extendible

A modular, extendible RAG (Retrieval-Augmented Generation) system using LangChain for answering questions about Chalmers courses. Includes a simple Gradio UI for interactive queries.

## Features

- **Modular Architecture**: Separated concerns with independent, testable components
- **Extendible Design**: Easy to add new LLM providers, vector stores, or features
- **Configuration-Driven**: All settings in YAML config file
- **Simple UI**: Gradio interface for interactive queries
- **Best Practices**: Follows LangChain patterns and Python best practices

## Architecture

```
┌─────────────┐
│   Config    │ ──> Centralized configuration
└──────┬──────┘
       │
       ├──> Vector Store ──> Retriever ──> RAG Chain ──> UI
       │
       ├──> LLM Factory ──> LLM ──────────┘
       │
       └──> Prompt Templates ────────────┘
```

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

### Setup Vector Database (One-Time)

**On Minerva:**
```bash
sbatch run_setup_langchain.sh
```

**Locally:**
```bash
python -m src.retrieval.langchain_setup
```

### Run Tests

**On Minerva:**
```bash
sbatch run_langchain_rag.sh
```

**Locally:**
```bash
python test_langchain_rag.py
```

### Launch UI

**On Minerva:**
```bash
sbatch run_ui_minerva.sh
# Then on your local machine:
# ssh -L 7860:compute-node:7860 minerva
# Open http://localhost:7860
```

**Locally:**
```bash
bash run_ui.sh
# Or:
python -m src.ui.gradio_app
```

## Project Structure

```
project_langchain/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── retrieval/
│   │   ├── vector_store.py      # Vector store factory
│   │   └── langchain_setup.py   # Setup script
│   ├── generation/
│   │   ├── llm_factory.py       # LLM factory (extendible)
│   │   ├── prompt_templates.py  # Prompt management
│   │   └── rag_chain.py         # RAG chain builder
│   ├── ui/
│   │   └── gradio_app.py        # Gradio UI
│   └── utils/
│       └── config_loader.py      # Config loader
├── data/                         # Symlink to project/data/
├── test_langchain_rag.py         # Test script
├── run_ui.sh                     # Local UI runner
├── run_ui_minerva.sh             # Minerva UI batch script
├── run_setup_langchain.sh        # Setup batch script
├── run_langchain_rag.sh          # Test batch script
└── requirements.txt
```

## Configuration

All settings are in `config/config.yaml`:

- **Model settings**: Model name, device, max tokens, temperature
- **Retrieval settings**: Top-k, embedding model
- **Vector store settings**: Collection name, persist directory
- **UI settings**: Port, share, title

Environment variables can override config:
- `HF_MODEL_NAME`: Override model name
- `HF_DEVICE`: Override device
- `GRADIO_PORT`: Override UI port
- `GRADIO_SHARE`: Enable/disable public sharing

## Module Descriptions

### Retrieval Module (`src/retrieval/`)

- **`vector_store.py`**: Factory for creating/loading ChromaDB vector stores
- **`langchain_setup.py`**: Converts courses to LangChain Documents and creates vector store

### Generation Module (`src/generation/`)

- **`llm_factory.py`**: Factory for creating LLMs (currently HuggingFace, easy to extend)
- **`prompt_templates.py`**: Centralized prompt template management
- **`rag_chain.py`**: Modular RAG chain builder

### UI Module (`src/ui/`)

- **`gradio_app.py`**: Gradio interface for interactive queries

### Utils Module (`src/utils/`)

- **`config_loader.py`**: Loads and validates configuration with environment variable support

## Extending the System

### Add a New LLM Provider

1. Add method to `LLMFactory` class in `src/generation/llm_factory.py`:
```python
@staticmethod
def create_openai_llm(api_key: str, model: str = "gpt-3.5-turbo"):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(openai_api_key=api_key, model_name=model)
```

2. Update `create()` method to handle new provider

3. Add configuration to `config/config.yaml`

### Add a New Vector Store

1. Add method to `VectorStoreFactory` class in `src/retrieval/vector_store.py`:
```python
@staticmethod
def create_pinecone_store(index_name: str, api_key: str):
    from langchain_pinecone import PineconeVectorStore
    # Implementation
```

2. Update `create_vector_store()` to support new store type

### Add a New Prompt Template

1. Add template string to `PromptTemplateManager` class in `src/generation/prompt_templates.py`

2. Add getter method or update `get_template()` method

### Add Conversation Memory

1. Extend `RAGChainBuilder` in `src/generation/rag_chain.py`:
```python
from langchain.memory import ConversationBufferMemory

def __init__(self, ..., memory: Optional[ConversationBufferMemory] = None):
    self.memory = memory
    # Update chain building to include memory
```

## Differences from Original Project

1. **LangChain Integration**: Uses LangChain abstractions instead of direct ChromaDB/transformers
2. **Modular Design**: Components are separated and easily swappable
3. **Configuration**: Centralized YAML config instead of hardcoded values
4. **UI Included**: Gradio interface for interactive queries
5. **Extensibility**: Factory patterns make it easy to add new providers

## Performance

- **Vector Database Setup**: ~5-10 minutes (one-time)
- **Model Loading**: ~1-2 minutes first time, ~10-30 seconds cached
- **Per Query**: ~0.5-2 seconds on GPU, ~10-30 seconds on CPU
- **UI Startup**: ~1-2 minutes (includes model loading)

## Troubleshooting

**Vector store not found:**
- Run setup script: `python -m src.retrieval.langchain_setup`

**Model access denied:**
- Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Make sure `HF_TOKEN` is set in `.env`

**UI not accessible on Minerva:**
- Check port forwarding: `ssh -L 7860:compute-node:7860 minerva`
- Verify job is running: `squeue -u $USER`
- Check logs: `tail -f ui_output_*.log`

## License

Same as parent project.

