# Phase 2: Retrieval System Setup Guide

## Overview

In Phase 2, you'll set up a semantic search system that can find relevant courses based on user queries.

## What You'll Build

1. **Vector Database** - Store course embeddings (ChromaDB)
2. **Embedding Model** - Convert text to vectors (sentence-transformers)
3. **Retrieval System** - Find similar courses based on queries

## Step-by-Step Instructions

### Step 1: Install Dependencies

```bash
pip install chromadb sentence-transformers
```

### Step 2: Run the Setup Script

```bash
python src/retrieval/setup_retrieval.py
```

This will:
- Load your cleaned course data
- Create embeddings for all 48 courses
- Store them in ChromaDB
- Test retrieval with sample queries

### Step 3: Test Your Own Queries

After setup, you can test retrieval:

```python
from src.retrieval.setup_retrieval import CourseRetriever

retriever = CourseRetriever(
    collection_name="chalmers_courses",
    persist_directory="data/chroma_db"
)

# Test a query
results = retriever.retrieve("machine learning courses", top_k=5)
for result in results:
    print(f"{result['course_code']}: {result['course_name']}")
```

## How It Works

1. **Document Preparation**: Each course is converted to a text document containing:
   - Course code and name
   - Course type
   - Full description

2. **Embedding**: The text is converted to a vector (512 dimensions) using `all-MiniLM-L6-v2`

3. **Storage**: Vectors are stored in ChromaDB with metadata

4. **Retrieval**: When you query, your question is also embedded, and ChromaDB finds the most similar course vectors

## Expected Output

When you run the setup script, you should see:

```
✅ Loaded 48 courses
✅ Created new collection: chalmers_courses
✅ Embedding model loaded
✅ Added 48 courses to vector database

Testing Retrieval System:
🔍 Query: What courses cover machine learning?
1. DAT441: Advanced topics in machine learning
2. DAT450: Machine learning for natural language processing
...
```

## Troubleshooting

### "Collection already exists"
- The script will ask if you want to rebuild
- Type 'y' to rebuild, 'n' to use existing

### Slow embedding generation
- This is normal - embedding 48 courses takes ~30 seconds
- The model downloads on first use (~80MB)

### No results found
- Check that courses were added successfully
- Try simpler queries like "machine learning"

## Next Phase

Once retrieval is working, move to **Phase 3: Generation System** where you'll:
- Integrate an LLM (OpenAI/Claude/local)
- Create prompts that use retrieved courses
- Generate answers based on course information

