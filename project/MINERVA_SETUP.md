# Running RAG Pipeline on Minerva

## Quick Setup

1. **Clone/Pull the repository:**
   ```bash
   cd /path/to/your/workspace
   git pull origin main
   cd project
   ```

2. **Create virtual environment (REQUIRED on Minerva):**
   ```bash
   # Create virtual environment
   python3 -m venv venv_minerva
   
   # Activate it
   source venv_minerva/bin/activate
   
   # Verify you're using the venv Python
   which python  # Should show: .../venv_minerva/bin/python
   ```

3. **Install dependencies (make sure venv is activated!):**
   ```bash
   # Make sure venv is activated (you should see (venv_minerva) in your prompt)
   pip install -r requirements.txt
   ```
   
   **Note:** If you get "externally-managed-environment" error, make sure:
   - You activated the virtual environment (`source venv_minerva/bin/activate`)
   - You're using `pip` from the venv (check with `which pip`)

4. **Set up your Hugging Face token:**
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_actual_token_here" > .env
   
   # Or use nano/vim to edit:
   nano .env
   ```
   
   Add your token:
   ```
   HF_TOKEN=hf_your_actual_token_here
   ```

5. **Request access to Llama 3.2 1B-Instruct:**
   - Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - Click "Agree and access repository"
   - Wait for approval (usually instant)

6. **Run the RAG test:**
   ```bash
   python test_rag.py
   ```

## Using SLURM (if needed)

If you want to run it as a batch job:

```bash
# Create a simple run script
cat > run_rag.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rag_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=8G

source venv_minerva/bin/activate
python test_rag.py
EOF

chmod +x run_rag.sh
sbatch run_rag.sh
```

## Files Needed

- `test_rag.py` - Main RAG proof of concept test
- `src/retrieval/setup_retrieval.py` - Retrieval system
- `src/generation/llama_generator.py` - Llama generator
- `data/chroma_db/` - Your vector database (should already exist)
- `.env` - Your Hugging Face token (create this, not in git)

## Notes

- The model will download ~2GB on first run (Llama 3.2 1B)
- It will be cached in `~/.cache/huggingface/` for future runs
- Make sure you have enough disk space (~3GB free recommended)

