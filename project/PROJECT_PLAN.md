# RAG-Based University Course Chatbot - Project Plan

## Project Overview

**Project Title:** Retrieval-Augmented Generation (RAG) Chatbot for University Course Information

**Team Size:** 4 members

**Objective:** Develop a chatbot that can answer questions about courses available at the university using Retrieval-Augmented Generation. The system will retrieve relevant course information from a knowledge base and generate accurate, context-aware responses.

---

## 1. Project Goals

### Primary Goals
- Build a functional RAG system that can answer questions about university courses
- Implement efficient document retrieval from course information
- Generate accurate and helpful responses using LLMs
- Create an intuitive user interface for interaction

### Success Criteria
- System can answer at least 80% of common course-related questions accurately
- Response time < 3 seconds for typical queries
- Retrieval precision > 0.7 (relevant documents in top-k results)
- User satisfaction score > 4/5

---

## 2. Team Roles & Responsibilities

### Team Member 1: Data Engineer & Knowledge Base Specialist
**Responsibilities:**
- Collect and preprocess course data (course descriptions, prerequisites, schedules, etc.)
- Design and implement document storage system (vector database)
- Create data pipelines for knowledge base updates
- Handle data cleaning and normalization

**Key Deliverables:**
- Course data collection scripts
- Vector database setup (e.g., ChromaDB, Pinecone, or FAISS)
- Data preprocessing pipeline

### Team Member 2: Retrieval System Developer
**Responsibilities:**
- Implement embedding models for document and query encoding
- Design and optimize retrieval algorithms (semantic search, hybrid search)
- Implement re-ranking mechanisms
- Optimize retrieval performance and accuracy

**Key Deliverables:**
- Embedding pipeline
- Retrieval system with multiple strategies
- Evaluation metrics for retrieval quality

### Team Member 3: Generation & LLM Integration Specialist
**Responsibilities:**
- Integrate LLM APIs (OpenAI, Anthropic, or local models)
- Design prompt engineering strategies for RAG
- Implement response generation with context injection
- Handle conversation management and context tracking

**Key Deliverables:**
- LLM integration module
- Prompt templates and optimization
- Response generation pipeline

### Team Member 4: Frontend & Evaluation Lead
**Responsibilities:**
- Develop user interface (web app or chatbot interface)
- Create evaluation framework and test datasets
- Implement user feedback collection
- Conduct end-to-end testing and performance evaluation

**Key Deliverables:**
- User interface (Streamlit/Gradio/React)
- Evaluation framework
- Test dataset and evaluation reports

---

## 3. Technical Architecture

### System Components

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Encoder  │ (Embedding Model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │ (ChromaDB/FAISS/Pinecone)
│  (Course Docs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │ (Top-k Similar Documents)
│   Engine        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context Builder│ (Format Retrieved Docs)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Generator │ (GPT-4/Claude/Local Model)
│   + Prompt      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Response      │
└─────────────────┘
```

### Technology Stack

**Data & Storage:**
- Vector Database: ChromaDB / FAISS / Pinecone
- Document Storage: JSON / SQLite / PostgreSQL
- Data Processing: pandas, numpy

**NLP & ML:**
- Embedding Models: sentence-transformers, OpenAI embeddings, or multilingual models
- LLM: OpenAI GPT-4/3.5, Anthropic Claude, or local models (Llama 2/3, Mistral)
- Framework: LangChain / LlamaIndex (optional, for RAG orchestration)

**Backend:**
- Python 3.9+
- FastAPI / Flask (for API if needed)

**Frontend:**
- Streamlit / Gradio (quick prototyping)
- React + FastAPI (for production-like UI)

**Evaluation:**
- RAGAS / custom evaluation metrics
- Test datasets with ground truth

---

## 4. Project Phases & Timeline

### Phase 1: Setup & Data Collection (Week 1-2)
**Duration:** 2 weeks

**Tasks:**
- [ ] Set up project repository and development environment
- [ ] Research and collect course data sources (university website, course catalogs, etc.)
- [ ] Design data schema for course information
- [ ] Implement data collection scripts
- [ ] Clean and normalize collected data
- [ ] Create initial knowledge base (100+ course documents)

**Deliverables:**
- Project repository with structure
- Course dataset (JSON/CSV format)
- Data collection documentation

---

### Phase 2: Retrieval System Development (Week 3-4)
**Duration:** 2 weeks

**Tasks:**
- [ ] Set up vector database
- [ ] Select and test embedding models
- [ ] Implement document chunking strategy
- [ ] Create document embedding pipeline
- [ ] Implement semantic search retrieval
- [ ] Test retrieval with sample queries
- [ ] Evaluate retrieval metrics (precision, recall, MRR)

**Deliverables:**
- Vector database with embedded documents
- Retrieval module with API
- Retrieval evaluation report

---

### Phase 3: Generation System Development (Week 5-6)
**Duration:** 2 weeks

**Tasks:**
- [ ] Set up LLM API access or local model
- [ ] Design prompt templates for RAG
- [ ] Implement context injection mechanism
- [ ] Build response generation pipeline
- [ ] Handle edge cases (no relevant docs, ambiguous queries)
- [ ] Implement conversation memory (optional)

**Deliverables:**
- Generation module
- Prompt templates and documentation
- Sample generated responses

---

### Phase 4: Integration & Frontend (Week 7-8)
**Duration:** 2 weeks

**Tasks:**
- [ ] Integrate retrieval and generation components
- [ ] Build user interface (Streamlit/Gradio)
- [ ] Implement error handling and logging
- [ ] Add user feedback mechanism
- [ ] End-to-end testing
- [ ] Performance optimization

**Deliverables:**
- Working RAG chatbot application
- User interface
- Integration documentation

---

### Phase 5: Evaluation & Refinement (Week 9-10)
**Duration:** 2 weeks

**Tasks:**
- [ ] Create evaluation test set (50-100 questions with ground truth)
- [ ] Implement evaluation metrics (accuracy, relevance, fluency)
- [ ] Conduct user testing (if possible)
- [ ] Analyze failure cases and improve system
- [ ] Optimize retrieval and generation parameters
- [ ] Prepare final presentation and documentation

**Deliverables:**
- Evaluation report with metrics
- Improved system based on feedback
- Final presentation materials
- Project documentation

---

## 5. Detailed Task Breakdown

### Data Collection Tasks
1. Identify course data sources (university website, course catalogs, syllabi)
2. Web scraping or API access setup
3. Extract: course name, code, description, prerequisites, credits, schedule, instructor
4. Data cleaning: remove HTML, normalize text, handle missing fields
5. Create structured dataset (JSON/CSV)
6. Validate data quality

### Retrieval System Tasks
1. Choose embedding model (e.g., `all-MiniLM-L6-v2`, `text-embedding-ada-002`)
2. Set up vector database
3. Implement chunking: split long documents into smaller chunks (overlap strategy)
4. Generate embeddings for all chunks
5. Store embeddings in vector database
6. Implement query encoding
7. Implement similarity search (cosine similarity)
8. Add re-ranking (optional: cross-encoder)
9. Evaluate retrieval quality

### Generation System Tasks
1. Choose LLM (OpenAI GPT-3.5/4, Claude, or local model)
2. Design prompt template:
   ```
   Context: [retrieved documents]
   Question: [user query]
   Answer: [generate based on context]
   ```
3. Implement context formatting
4. Handle token limits (truncate if needed)
5. Implement response post-processing
6. Add source citation (which courses/documents were used)

### Frontend Tasks
1. Design UI mockup
2. Implement chat interface
3. Display responses with formatting
4. Show source documents/references
5. Add loading indicators
6. Implement error messages
7. Add user feedback buttons (thumbs up/down)

### Evaluation Tasks
1. Create test question set (various question types):
   - Factual: "What are the prerequisites for course X?"
   - Comparison: "What's the difference between course A and B?"
   - Recommendation: "What courses should I take for machine learning?"
   - Complex: "Can I take course X if I've completed Y but not Z?"
2. Define evaluation metrics:
   - Retrieval: Precision@k, Recall@k, MRR
   - Generation: BLEU, ROUGE, semantic similarity
   - End-to-end: Answer accuracy (human evaluation)
3. Run evaluation on test set
4. Analyze results and identify improvements

---

## 6. Knowledge Base Design

### Course Document Schema
```json
{
  "course_code": "DAT450",
  "course_name": "Natural Language Processing",
  "credits": 7.5,
  "description": "Course description text...",
  "prerequisites": ["DAT250", "DAT260"],
  "learning_outcomes": ["...", "..."],
  "schedule": "Autumn 2025",
  "instructor": "Dr. X",
  "department": "Computer Science",
  "level": "Master's"
}
```

### Chunking Strategy
- Chunk size: 200-500 tokens
- Overlap: 50-100 tokens
- Metadata: Include course_code, course_name in each chunk

---

## 7. Evaluation Framework

### Metrics

**Retrieval Metrics:**
- Precision@k: Proportion of retrieved docs that are relevant
- Recall@k: Proportion of relevant docs retrieved
- Mean Reciprocal Rank (MRR): Average of 1/rank of first relevant doc

**Generation Metrics:**
- Answer Accuracy: Human evaluation (correct/incorrect)
- Semantic Similarity: Compare generated answer to ground truth
- Fluency: Naturalness of language
- Citation Accuracy: Are sources correctly referenced?

**System Metrics:**
- Response Time: End-to-end latency
- Throughput: Queries per second
- Cost: API costs per query (if using paid APIs)

### Test Dataset
Create 50-100 test questions covering:
- Simple factual queries (30%)
- Complex multi-hop queries (30%)
- Comparison queries (20%)
- Recommendation queries (20%)

---

## 8. Risk Management

### Technical Risks
1. **Poor retrieval quality**
   - Mitigation: Test multiple embedding models, implement hybrid search
   
2. **LLM API costs/limits**
   - Mitigation: Use local models as backup, implement caching

3. **Data quality issues**
   - Mitigation: Implement data validation, manual review of samples

4. **Integration complexity**
   - Mitigation: Use frameworks like LangChain, modular design

### Project Risks
1. **Scope creep**
   - Mitigation: Stick to core features, document scope clearly

2. **Team coordination**
   - Mitigation: Regular meetings, clear task assignments, shared documentation

3. **Timeline delays**
   - Mitigation: Buffer time in schedule, prioritize core features

---

## 9. Deliverables

### Code Deliverables
- [ ] Data collection and preprocessing scripts
- [ ] Vector database setup and embedding pipeline
- [ ] Retrieval system module
- [ ] Generation system module
- [ ] Integrated RAG application
- [ ] User interface
- [ ] Evaluation framework and scripts

### Documentation Deliverables
- [ ] Project README with setup instructions
- [ ] Architecture documentation
- [ ] API documentation (if applicable)
- [ ] Evaluation report
- [ ] User guide
- [ ] Final presentation slides

### Data Deliverables
- [ ] Course knowledge base (structured dataset)
- [ ] Test evaluation dataset
- [ ] Evaluation results and analysis

---

## 10. Meeting Schedule

**Weekly Team Meetings:**
- Time: [To be decided]
- Format: 30-60 minutes
- Agenda: Progress updates, blockers, next steps

**Sprint Reviews:**
- End of each phase
- Demo working components
- Review and adjust plan

---

## 11. Next Steps

1. **Immediate Actions:**
   - Assign team roles (if not already done)
   - Set up shared repository (Git)
   - Schedule first team meeting
   - Research course data sources

2. **Week 1 Goals:**
   - Complete project setup
   - Begin data collection
   - Finalize technical stack decisions

---

## 12. Resources & References

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "In-Context Retrieval-Augmented Language Models" (Ram et al., 2023)

### Tools & Libraries
- LangChain: https://python.langchain.com/
- LlamaIndex: https://www.llamaindex.ai/
- ChromaDB: https://www.trychroma.com/
- RAGAS: https://github.com/explodinggradients/ragas

### Datasets
- University course catalogs
- Course syllabi (if available)

---

## Appendix: Project Structure

```
project/
├── data/
│   ├── raw/              # Raw course data
│   ├── processed/        # Cleaned and structured data
│   └── test_queries.json # Evaluation test set
├── src/
│   ├── data_collection/  # Data scraping/collection scripts
│   ├── preprocessing/    # Data cleaning and preparation
│   ├── retrieval/        # Retrieval system
│   ├── generation/       # LLM integration and generation
│   ├── evaluation/       # Evaluation framework
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for exploration
├── frontend/            # User interface code
├── tests/               # Unit and integration tests
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
├── README.md           # Project overview
└── PROJECT_PLAN.md     # This file
```

---

**Last Updated:** [Date]
**Project Status:** Planning Phase
**Team Members:** [Names to be added]

