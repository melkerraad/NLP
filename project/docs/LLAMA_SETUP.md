# Llama 3.2 3B Setup Guide

This guide will help you set up Llama 3.2 3B for local RAG generation.

## Prerequisites

1. **GPU Access** (recommended) or CPU (slower)
2. **Python 3.9+**
3. **Hugging Face Account** (free)

## Step 1: Get Hugging Face Token

1. Go to https://huggingface.co/join and create a free account
2. Go to https://huggingface.co/settings/tokens
3. Create a new token (read access is enough)
4. Copy the token

## Step 2: Set Environment Variable

### Windows (PowerShell)
```powershell
$env:HF_TOKEN="your_token_here"
```

### Windows (Command Prompt)
```cmd
set HF_TOKEN=your_token_here
```

### Linux/Mac
```bash
export HF_TOKEN="your_token_here"
```

**Or create a `.env` file in the project root:**
```
HF_TOKEN=your_token_here
```

## Step 3: Install Dependencies

```bash
pip install transformers accelerate torch sentencepiece
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Step 4: Test the Setup

Run the test script:
```bash
python test_rag_full.py
```

On first run, the model will be downloaded (~6GB). This may take a few minutes.

## Usage

### Basic Usage

```python
from src.generation.llama_generator import LlamaRAGGenerator
from src.retrieval.setup_retrieval import CourseRetriever

# Initialize retriever
retriever = CourseRetriever(
    collection_name="chalmers_courses",
    persist_directory="data/chroma_db"
)

# Initialize generator
generator = LlamaRAGGenerator()

# Retrieve courses
query = "What courses teach deep learning?"
retrieved = retriever.retrieve(query, top_k=3)

# Format for generator
context_docs = [
    {
        'course_code': r['course_code'],
        'course_name': r['course_name'],
        'document': r['document']
    }
    for r in retrieved
]

# Generate response
result = generator.generate_with_sources(query, context_docs)
print(result['answer'])
print(f"Sources: {result['sources']}")
```

### Advanced Options

```python
# Use CPU instead of GPU
generator = LlamaRAGGenerator(device="cpu")

# Custom model (if you have access)
generator = LlamaRAGGenerator(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda"
)

# Custom generation parameters
answer = generator.generate(
    query,
    context_docs,
    max_new_tokens=1000,  # Longer responses
    temperature=0.5,       # More focused (lower = more deterministic)
    top_p=0.95            # Nucleus sampling
)
```

## Performance Tips

1. **Use GPU**: Much faster (~10x) than CPU
2. **First run**: Model download takes time (~6GB)
3. **Subsequent runs**: Model loads from cache (faster)
4. **Memory**: Needs ~6GB RAM/VRAM for 3B model

## Troubleshooting

### "No Hugging Face token found"
- Set `HF_TOKEN` environment variable
- Or pass `hf_token` parameter to `LlamaRAGGenerator()`

### "CUDA out of memory"
- Use smaller model or CPU: `LlamaRAGGenerator(device="cpu")`
- Reduce `max_new_tokens`

### "Model not found"
- Make sure you've accepted the model license at:
  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Check your Hugging Face token is valid

### Slow generation
- Make sure GPU is being used (check `device` in output)
- GPU should show "cuda" not "cpu"
- First generation is slower (model loading)

## Model Information

- **Model**: Llama 3.2 3B Instruct
- **Size**: ~6GB
- **License**: Llama 3 Community License (free for research/education)
- **Performance**: Good for RAG tasks, follows instructions well
- **Speed**: ~1-2 seconds per response on GPU, ~5-10 seconds on CPU

## Alternative Models

If you want to try other models:

```python
# Mistral 7B (better quality, needs more memory)
generator = LlamaRAGGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)

# Llama 3.1 8B (best quality, needs ~16GB)
generator = LlamaRAGGenerator(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda"
)
```

