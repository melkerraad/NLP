#!/bin/bash
# Run Gradio UI locally

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "venv_minerva" ]; then
    source venv_minerva/bin/activate
fi

# Set working directory
cd "$(dirname "$0")"

# Run UI
echo "Starting RAG Chatbot UI..."
python -m src.ui.gradio_app

