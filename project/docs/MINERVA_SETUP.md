# Running RAG System on Minerva

This guide explains how to run the RAG course chatbot on Chalmers Minerva compute cluster.

## Quick Start

```bash
# 1. SSH to Minerva
ssh your_username@minerva.chalmers.se

# 2. Clone/navigate to your project
cd ~/workspace/your_repo/project

# 3. Run setup (first time only)
bash setup_minerva.sh

# 4. Set Hugging Face token
export HF_TOKEN="your_token_here"

# 5. Submit job
sbatch run_minerva.sh

# 6. Check status
squeue -u $USER
```

## Detailed Setup

### Step 1: Transfer Code to Minerva

#### Option A: Git Clone (Recommended)

```bash
# SSH into Minerva
ssh your_username@minerva.chalmers.se

# Navigate to workspace
cd ~/workspace  # or your preferred location

# Clone repository
git clone https://git.chalmers.se/your_username/your_repo.git
cd your_repo/project
```

#### Option B: SCP Transfer

```bash
# From your local machine
scp -r project/ your_username@minerva.chalmers.se:~/workspace/
```

### Step 2: Initial Setup

Run the setup script:

```bash
bash setup_minerva.sh
```

This will:
- Load required modules (Python, CUDA)
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies

### Step 3: Configure Environment

#### Set Hugging Face Token

```bash
# For current session
export HF_TOKEN="your_token_here"

# For persistence (add to ~/.bashrc)
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Transfer Data (if needed)

```bash
# From local machine
scp -r project/data/ your_username@minerva.chalmers.se:~/workspace/your_repo/project/
```

Or generate on Minerva:

```bash
# Set up retrieval database
python -m src.retrieval.setup_retrieval
```

### Step 4: Run Jobs

#### Option A: Submit SLURM Job (Recommended)

```bash
# Submit job
sbatch run_minerva.sh

# Check job status
squeue -u $USER

# View output (once job starts)
tail -f rag_output_JOBID.log
```

#### Option B: Interactive Session (For Testing)

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash

# Once you get a node
source venv_minerva/bin/activate
export HF_TOKEN="your_token"
python test_rag_full.py
```

## SLURM Job Script Details

The `run_minerva.sh` script includes:

- **Partition**: `gpu` - GPU partition
- **GPU**: `--gres=gpu:1` - Request 1 GPU
- **Memory**: `16G` - 16GB RAM
- **Time**: `02:00:00` - 2 hours max
- **Output**: Logs saved to `rag_output_JOBID.log`

### Customize Job Parameters

Edit `run_minerva.sh`:

```bash
# More memory
#SBATCH --mem=32G

# Longer time
#SBATCH --time=04:00:00

# Multiple GPUs
#SBATCH --gres=gpu:2
```

## Using GPU Models

The code automatically detects GPU. To force GPU usage:

```python
from src.generation.llama_generator import LlamaRAGGenerator

# Will use GPU if available
generator = LlamaRAGGenerator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",  # Once approved
    device="cuda"  # Force GPU
)
```

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA module loaded
module list
```

### Out of Memory

- Reduce `max_new_tokens` in generation
- Use smaller model
- Request more memory: `--mem=32G`

### Module Errors

```bash
# Check available modules
module avail Python
module avail CUDA

# Load correct versions
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
```

### Disk Space Issues

```bash
# Use scratch space for cache
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface

# Or clear cache
rm -rf ~/.cache/huggingface/hub
```

### Virtual Environment Issues

```bash
# Recreate if needed
rm -rf venv_minerva
bash setup_minerva.sh
```

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch Processing**: Process multiple queries together
3. **Model Caching**: Models cached after first download
4. **Scratch Space**: Use `/scratch/$USER/` for large files

## Example Workflow

```bash
# 1. SSH to Minerva
ssh username@minerva.chalmers.se

# 2. Navigate to project
cd ~/workspace/your_repo/project

# 3. Activate environment
source venv_minerva/bin/activate

# 4. Set token
export HF_TOKEN="your_token"

# 5. Test interactively
srun --partition=gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
python test_rag_full.py

# 6. Or submit job
sbatch run_minerva.sh
```

## Monitoring Jobs

```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job JOBID

# View live output
tail -f rag_output_JOBID.log

# Cancel job
scancel JOBID
```

## Next Steps

Once running on Minerva:
1. ✅ Test with Llama 3.2 3B (once access approved)
2. Build frontend interface
3. Run evaluation benchmarks
4. Optimize for production

## Resources

- Minerva Documentation: Check Chalmers internal docs
- SLURM Guide: `man sbatch`, `man srun`
- GPU Monitoring: `nvidia-smi`, `watch -n 1 nvidia-smi`
