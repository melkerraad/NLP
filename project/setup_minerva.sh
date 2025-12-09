#!/bin/bash
# Setup script for Minerva cluster
# Run this once to set up your environment

echo "=== Minerva Setup Script ==="
echo ""

# Load modules
echo "Loading modules..."
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0

# Check GPU
echo ""
echo "=== GPU Check ==="
nvidia-smi

# Create virtual environment
echo ""
echo "=== Creating Virtual Environment ==="
python -m venv venv_minerva
source venv_minerva/bin/activate

# Upgrade pip
echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other requirements
echo ""
echo "=== Installing Requirements ==="
pip install -r requirements.txt

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Set HF_TOKEN: export HF_TOKEN='your_token'"
echo "2. Transfer data files if needed"
echo "3. Run: sbatch run_minerva.sh"

