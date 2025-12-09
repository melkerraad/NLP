# Project Decisions Log

This document tracks major technical and design decisions made during the project.

## Format
Each decision should include:
- **Date**: When the decision was made
- **Decision**: What was decided
- **Rationale**: Why this decision was made
- **Alternatives Considered**: Other options that were discussed
- **Impact**: How this affects the project

---

## Decision 1: Vector Database Choice
**Date:** [To be filled]
**Decision:** [Chosen database: ChromaDB/FAISS/Pinecone]
**Rationale:** 
**Alternatives Considered:**
**Impact:**

---

## Decision 2: Embedding Model Selection
**Date:** [To be filled]
**Decision:** [Chosen model]
**Rationale:**
**Alternatives Considered:**
**Impact:**

---

## Decision 3: LLM Provider Selection
**Date:** 2025-01-XX
**Decision:** Llama 3.2 3B Instruct (local, free, open source)
**Rationale:** 
- Completely free and open source
- Good quality for RAG tasks
- Runs locally (privacy, no API costs)
- GPU access available for good performance
- No API rate limits or costs
**Alternatives Considered:**
- OpenAI GPT-3.5-turbo: Excellent quality but requires API key and costs money
- Mistral 7B: Better quality but requires more GPU memory
- GPT-4: Best quality but expensive and requires API
**Impact:**
- Requires GPU setup (or CPU, slower)
- Need Hugging Face token for model access
- Model download ~6GB on first run
- Local generation gives full control and privacy

---

## Decision 4: Frontend Framework
**Date:** [To be filled]
**Decision:** [Streamlit/Gradio/React]
**Rationale:**
**Alternatives Considered:**
**Impact:**

---

## Decision 5: Chunking Strategy
**Date:** [To be filled]
**Decision:** [Chunk size and overlap]
**Rationale:**
**Alternatives Considered:**
**Impact:**

---

[Add more decisions as the project progresses]

