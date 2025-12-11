#!/bin/bash
#SBATCH --job-name=setup_langchain_db
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem=4G
#SBATCH --output=setup_db_output_%j.log
#SBATCH --error=setup_db_error_%j.log

# Note: This script populates the LangChain vector database with course embeddings
# GPU is optional - SentenceTransformer works fine on CPU for this task

# Activate virtual environment
if [ -d "venv_minerva" ]; then
    source venv_minerva/bin/activate
else
    echo "ERROR: venv_minerva not found!"
    echo "Create it first: python3 -m venv venv_minerva"
    exit 1
fi

# Set working directory
cd $SLURM_SUBMIT_DIR

# Print Python info
echo "=== Environment Information ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run the setup script
echo "=== Starting LangChain Database Setup ==="
python -m src.retrieval.langchain_setup

echo ""
echo "=== Job Complete ==="

