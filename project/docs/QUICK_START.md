# Quick Start Guide

## Initial Setup

1. **Clone/Navigate to project folder**
   ```bash
   cd project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file)
   ```env
   OPENAI_API_KEY=your_key_here  # If using OpenAI
   ANTHROPIC_API_KEY=your_key_here  # If using Claude
   ```

## Development Workflow

### For Data Collection Team Member
1. Start in `src/data_collection/`
2. Create scripts to scrape/collect course data
3. Save raw data to `data/raw/`
4. Document data sources and collection methods

### For Retrieval Team Member
1. Start in `src/retrieval/`
2. Set up vector database (ChromaDB recommended for start)
3. Implement embedding pipeline
4. Test retrieval with sample queries

### For Generation Team Member
1. Start in `src/generation/`
2. Set up LLM API access
3. Design prompt templates
4. Test generation with sample contexts

### For Frontend Team Member
1. Start in `frontend/`
2. Create Streamlit/Gradio app
3. Integrate with retrieval and generation modules
4. Add UI components

## Testing Your Setup

### Test Vector Database
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("test")
collection.add(
    documents=["This is a test document"],
    ids=["1"]
)
results = collection.query(query_texts=["test"], n_results=1)
print(results)
```

### Test Embeddings
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode("Test sentence")
print(embeddings.shape)
```

### Test LLM (OpenAI example)
```python
from openai import OpenAI
client = OpenAI(api_key="your_key")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Next Steps

1. Review `PROJECT_PLAN.md` for detailed timeline
2. Assign team roles
3. Start Phase 1: Data Collection
4. Set up weekly team meetings

