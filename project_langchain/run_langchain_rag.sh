#!/bin/bash
#SBATCH --job-name=langchain_rag
#SBATCH --partition=short
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=rag_output_%j.log
#SBATCH --error=rag_error_%j.log

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

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
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run the RAG test
echo "=== Starting RAG Test ==="
python test_langchain_rag.py

echo ""
echo "=== Job Complete ==="

