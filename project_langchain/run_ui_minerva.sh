#!/bin/bash
#SBATCH --job-name=rag_ui
#SBATCH --partition=short
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=ui_output_%j.log
#SBATCH --error=ui_error_%j.log

# Run Gradio UI on Minerva with port forwarding
# After job starts, use SSH port forwarding:
# ssh -L 7860:compute-node:7860 minerva
# Then open http://localhost:7860 in your browser

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

# Get compute node hostname
COMPUTE_NODE=$(hostname)
echo "=== Port Forwarding Instructions ==="
echo "Run this command on your local machine:"
echo "  ssh -L 7860:${COMPUTE_NODE}:7860 minerva"
echo ""
echo "Then open http://localhost:7860 in your browser"
echo ""

# Run UI
echo "=== Starting RAG Chatbot UI ==="
python -m src.ui.gradio_app

echo ""
echo "=== Job Complete ==="

