# LLM Recommendations for RAG Course Chatbot

## Quick Answer

**For a university project with budget constraints:**
- **Best Free Option:** **Llama 3.2 3B** (via Hugging Face) - Good balance of quality and speed
- **Best Paid Option:** **OpenAI GPT-3.5-turbo** - Excellent quality, very affordable (~$0.50 per 1M tokens)
- **Best Open Source:** **Mistral 7B** or **Llama 3.1 8B** - Best quality for local models

**TL;DR:** Start with **GPT-3.5-turbo** (free tier available, very cheap). If you need completely free, use **Llama 3.2 3B** locally.

---

## Detailed Recommendations

### 🆓 Free & Open Source Options

#### 1. **Llama 3.2 3B** ⭐ **RECOMMENDED FOR FREE**
- **Provider:** Meta (via Hugging Face)
- **Cost:** Free
- **Quality:** Good for RAG tasks (follows instructions well)
- **Speed:** Fast (~1-2 seconds on CPU, <1s on GPU)
- **Requirements:** 
  - ~6GB RAM
  - Works on CPU (slower) or GPU (faster)
  - Requires Hugging Face token (free)
- **Pros:**
  - ✅ Completely free
  - ✅ Good instruction following
  - ✅ Fast inference
  - ✅ No API limits
  - ✅ Privacy (runs locally)
- **Cons:**
  - ❌ Requires GPU for best performance
  - ❌ Slightly lower quality than larger models
  - ❌ Need to manage model loading

**Setup:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Uses GPU if available
)
```

#### 2. **Mistral 7B Instruct**
- **Provider:** Mistral AI (via Hugging Face)
- **Cost:** Free
- **Quality:** Excellent (better than Llama 3.2 3B)
- **Speed:** Moderate (~2-3 seconds on GPU)
- **Requirements:** ~14GB RAM/VRAM
- **Pros:**
  - ✅ Better quality than 3B models
  - ✅ Great for RAG
  - ✅ Free and open source
- **Cons:**
  - ❌ Requires more memory
  - ❌ Slower than smaller models

#### 3. **Llama 3.1 8B Instruct**
- **Provider:** Meta (via Hugging Face)
- **Cost:** Free
- **Quality:** Excellent
- **Speed:** Moderate (~2-3 seconds on GPU)
- **Requirements:** ~16GB RAM/VRAM
- **Pros:**
  - ✅ Very high quality
  - ✅ Great instruction following
- **Cons:**
  - ❌ Requires significant memory
  - ❌ May be overkill for simple RAG tasks

#### 4. **Google Gemma 2B**
- **Provider:** Google (via Hugging Face)
- **Cost:** Free
- **Quality:** Good (smaller but efficient)
- **Speed:** Very fast
- **Requirements:** ~4GB RAM
- **Pros:**
  - ✅ Very lightweight
  - ✅ Fast inference
- **Cons:**
  - ❌ Lower quality than 3B+ models

---

### 💰 Paid API Options (Very Affordable)

#### 1. **OpenAI GPT-3.5-turbo** ⭐ **RECOMMENDED FOR PAID**
- **Cost:** ~$0.50 per 1M input tokens, $1.50 per 1M output tokens
- **Quality:** Excellent for RAG
- **Speed:** Very fast (~0.5-1 second)
- **Free Tier:** $5 free credit (enough for ~10,000 queries)
- **Pros:**
  - ✅ Best quality/price ratio
  - ✅ Very fast
  - ✅ Easy to use
  - ✅ Reliable API
  - ✅ Free tier available
- **Cons:**
  - ❌ Requires API key
  - ❌ Costs money after free tier
  - ❌ Data sent to OpenAI

**Cost Estimate for Your Project:**
- ~100 tokens per query (context + question)
- ~200 tokens per response
- **~$0.0003 per query** (very cheap!)
- Free tier ($5) = ~16,000 queries

#### 2. **OpenAI GPT-4o-mini**
- **Cost:** ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- **Quality:** Excellent (better than GPT-3.5)
- **Speed:** Fast (~1 second)
- **Pros:**
  - ✅ Better quality than GPT-3.5
  - ✅ Cheaper than GPT-4
- **Cons:**
  - ❌ Still costs money
  - ❌ May be overkill for RAG

#### 3. **Anthropic Claude 3 Haiku**
- **Cost:** ~$0.25 per 1M input tokens, $1.25 per 1M output tokens
- **Quality:** Excellent
- **Speed:** Fast
- **Pros:**
  - ✅ High quality
  - ✅ Good for longer contexts
- **Cons:**
  - ❌ More expensive than GPT-3.5
  - ❌ No free tier

#### 4. **Google Gemini Flash**
- **Cost:** Free tier available, then ~$0.075 per 1M tokens
- **Quality:** Good
- **Speed:** Fast
- **Pros:**
  - ✅ Very cheap
  - ✅ Free tier available
- **Cons:**
  - ❌ Slightly lower quality than GPT-3.5

---

## Recommendation by Use Case

### 🎓 **For Academic Project (Budget-Conscious)**
1. **Start with GPT-3.5-turbo** (free tier)
   - Use free $5 credit
   - If you need more, it's very cheap
   - Best quality/effort ratio

2. **If completely free is required:**
   - Use **Llama 3.2 3B** locally
   - Works well for RAG tasks
   - Requires GPU for best performance

### 🚀 **For Best Performance**
- **GPT-3.5-turbo** (API) - Fastest and easiest
- **Mistral 7B** (local) - Best free quality

### 🔒 **For Privacy/Offline**
- **Llama 3.2 3B** or **Mistral 7B** (local)
- No data leaves your machine

---

## Implementation Examples

### Option 1: OpenAI GPT-3.5-turbo (Recommended)
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(query, context_docs):
    context = "\n\n".join([doc['document'] for doc in context_docs])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful course advisor."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content
```

### Option 2: Llama 3.2 3B (Free, Local)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, query, context_docs):
        context = "\n\n".join([doc['document'] for doc in context_docs])
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful course advisor. Answer based on the context provided.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=500, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Option 3: Using LangChain (Easier Integration)
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# For OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# For local Llama (via HuggingFace)
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    device=0,  # GPU
    max_new_tokens=500
)
llm = HuggingFacePipeline(pipeline=pipe)
```

---

## Cost Comparison

| Model | Cost per 1K queries* | Quality | Speed |
|-------|---------------------|---------|-------|
| GPT-3.5-turbo | $0.30 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-4o-mini | $0.15 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Llama 3.2 3B (local) | $0 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mistral 7B (local) | $0 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Claude Haiku | $0.50 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

*Assuming ~300 tokens per query (context + question + response)

---

## Final Recommendation

**For your project, I recommend:**

1. **Primary:** Start with **GPT-3.5-turbo**
   - Use free tier ($5 credit)
   - Very cheap if you exceed (~$0.30 per 1000 queries)
   - Best quality and easiest setup

2. **Backup:** Set up **Llama 3.2 3B** locally
   - Completely free
   - Good quality for RAG
   - Useful if API limits are hit

3. **If you have GPU:** Consider **Mistral 7B**
   - Better quality than Llama 3.2 3B
   - Still free and local

**Why not completely free open source?**
- Open source is great, but requires:
  - GPU setup (if you want good performance)
  - More code to manage
  - Slower development
- GPT-3.5-turbo is so cheap (~$0.30 per 1000 queries) that it's worth it for the convenience
- You can always switch to local models later

---

## Next Steps

1. **Try GPT-3.5-turbo first** (easiest, free tier available)
2. **If budget is zero:** Set up Llama 3.2 3B locally
3. **Test both** and see which works better for your use case
4. **Document your choice** in `docs/DECISIONS.md`

