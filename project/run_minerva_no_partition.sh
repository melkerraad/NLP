#!/bin/bash
#SBATCH --job-name=rag_course_chatbot
# Alternative version without partition specification (uses default)
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=rag_output_%j.log
#SBATCH --error=rag_error_%j.log

# Load required modules
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Activate virtual environment (adjust path as needed)
if [ -d "venv_minerva" ]; then
    source venv_minerva/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Hugging Face token (or load from environment)
# export HF_TOKEN="your_token_here"

# Set working directory
cd $SLURM_SUBMIT_DIR

# Print Python and CUDA info
echo "=== Environment Information ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run the RAG test
echo "=== Starting RAG Test ==="
python test_rag_full.py

echo ""
echo "=== Job Complete ==="

