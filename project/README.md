# RAG-Based University Course Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions about university courses using semantic search and large language models.

## Project Overview

This project implements a chatbot that can answer questions about courses available at the university. The system uses RAG to retrieve relevant course information from a knowledge base and generate accurate, context-aware responses.

## Team Members

- [Member 1: Data Engineer & Knowledge Base Specialist]
- [Member 2: Retrieval System Developer]
- [Member 3: Generation & LLM Integration Specialist]
- [Member 4: Frontend & Evaluation Lead]

## Project Structure

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
├── README.md           # This file
└── PROJECT_PLAN.md     # Detailed project plan
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. Clone the repository (or navigate to project folder)
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. Collect course data (see `src/data_collection/`)
2. Set up vector database (see `src/retrieval/`)
3. Configure LLM API keys (if using external APIs)
4. Run the application (see `frontend/`)

## Usage

[To be updated as development progresses]

## Evaluation

Run evaluation scripts:
```bash
python src/evaluation/evaluate.py
```

## Documentation

See `PROJECT_PLAN.md` for detailed project plan and `docs/` for additional documentation.

## License

[To be determined]

