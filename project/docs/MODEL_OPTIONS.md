# Model Options for RAG Generation

## Recommended: Request Llama Access

**Best option:** Request access to Llama 3.2 3B at:
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

- Usually instant approval for academic use
- Best quality for RAG
- ~6GB model size

## Alternative Models

If you can't use Llama, here are alternatives:

### 1. Qwen2.5 0.5B (Smallest - ~1GB)
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Size:** ~1GB
- **Quality:** Good for simple RAG tasks
- **Speed:** Fast
- **Status:** Not gated, free to use

### 2. Phi-3 Mini (Medium - ~5GB)
- **Model:** `microsoft/Phi-3-mini-4k-instruct`
- **Size:** ~5GB
- **Quality:** Very good for RAG
- **Speed:** Fast
- **Status:** Not gated, free to use
- **Note:** Requires ~5GB free disk space

### 3. Mistral 7B (Largest - ~14GB)
- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Size:** ~14GB
- **Quality:** Excellent
- **Speed:** Moderate
- **Status:** Not gated, free to use
- **Note:** Requires significant disk space and GPU memory

## Using a Different Model

You can specify a different model when initializing:

```python
from src.generation.llama_generator import LlamaRAGGenerator

# Use Qwen (smallest)
generator = LlamaRAGGenerator(
    model_name="Qwen/Qwen2.5-0.5B-Instruct"
)

# Use Phi-3 Mini (better quality, needs more space)
generator = LlamaRAGGenerator(
    model_name="microsoft/Phi-3-mini-4k-instruct"
)

# Use Mistral 7B (best quality, needs GPU and lots of space)
generator = LlamaRAGGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)
```

## Disk Space Issues

If you're running out of disk space:

1. **Clear Hugging Face cache:**
   ```python
   from huggingface_hub import scan_cache_dir
   delete_strategy = scan_cache_dir().delete_revisions("Qwen/Qwen2.5-0.5B-Instruct")
   delete_strategy.execute()
   ```

2. **Or manually delete cache:**
   - Windows: `C:\Users\<username>\.cache\huggingface\hub\`
   - Linux/Mac: `~/.cache/huggingface/hub/`

3. **Use smaller model:** Qwen2.5 0.5B only needs ~1GB

## Recommendation

1. **First choice:** Request Llama 3.2 3B access (best quality)
2. **If no access:** Use Qwen2.5 0.5B (smallest, works well)
3. **If you have space:** Use Phi-3 Mini (better quality than Qwen)

