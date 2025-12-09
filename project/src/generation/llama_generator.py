"""Llama 3.2 3B local generation system for RAG responses.

This module uses Llama 3.2 3B Instruct model running locally with GPU support.
"""

from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path


class LlamaRAGGenerator:
    """Generation system for RAG responses using Llama 3.2 3B locally."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_alternative_if_gated: bool = True
    ):
        """Initialize the generator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            hf_token: Hugging Face token (required for Llama models)
            use_alternative_if_gated: If True, use non-gated alternative if access denied
        """
        print(f"Loading model: {model_name}...")
        print("This may take a few minutes on first run (downloading model)...")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Get Hugging Face token from environment if not provided
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        if hf_token is None:
            print("⚠️  Warning: No Hugging Face token found.")
            print("   Set HF_TOKEN environment variable or pass hf_token parameter.")
            print("   You can get a free token at: https://huggingface.co/settings/tokens")
        
        # Try to load model, fallback to alternative if gated
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=hf_token
            )
        except Exception as e:
            if "gated" in str(e).lower() or "403" in str(e) or "access" in str(e).lower():
                if use_alternative_if_gated:
                    print(f"\n⚠️  Access denied to {model_name}")
                    print("   Trying smaller alternative: Qwen/Qwen2.5-0.5B-Instruct (~1GB)")
                    print("   (To use Llama, request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)")
                    print("   (For better quality, try: microsoft/Phi-3-mini-4k-instruct - needs ~5GB)\n")
                    
                    # Use Qwen2.5 0.5B as alternative (smaller, ~1GB, not gated)
                    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                    self.model_name = model_name
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None
                    )
                else:
                    raise Exception(
                        f"Access denied to {model_name}. "
                        f"Request access at: https://huggingface.co/{model_name}"
                    ) from e
            else:
                raise
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ Model loaded on {device}")
        print(f"   Model: {model_name}")
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt with context and query in Llama 3.2 format.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        # Format context documents
        context_parts = []
        for doc in context_docs:
            course_code = doc.get('course_code', 'Unknown')
            course_name = doc.get('course_name', '')
            document = doc.get('document', '')
            
            context_parts.append(
                f"Course: {course_code} - {course_name}\n"
                f"Content: {document}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build prompt - adapt format based on model
        if "phi-3" in self.model_name.lower():
            # Phi-3 format
            prompt = f"""<|system|>
You are a helpful assistant that answers questions about university courses. Use the following course information to answer the question. If the information is not available in the context, say so.<|end|>
<|user|>
Course Information:
{context_text}

Question: {query}<|end|>
<|assistant|>
"""
        elif "qwen" in self.model_name.lower():
            # Qwen format
            prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions about university courses. Use the following course information to answer the question. If the information is not available in the context, say so.<|im_end|>
<|im_start|>user
Course Information:
{context_text}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Llama 3.2 format
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions about university courses. Use the following course information to answer the question. If the information is not available in the context, say so.<|eot_id|><|start_header_id|>user<|end_header_id|>

Course Information:
{context_text}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict],
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate response using Llama model.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response string
        """
        # Build prompt
        prompt = self.build_prompt(query, context_docs)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            # Handle pad_token for different models
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=pad_token_id
            )
        
        # Decode response (only the newly generated part)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_with_sources(
        self,
        query: str,
        context_docs: List[Dict],
        max_new_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict:
        """Generate response with source citations.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        answer = self.generate(query, context_docs, max_new_tokens, temperature)
        
        # Extract source course codes
        sources = []
        for doc in context_docs:
            course_code = doc.get('course_code', 'Unknown')
            if course_code and course_code != 'Unknown':
                sources.append(course_code)
        
        return {
            'answer': answer,
            'sources': list(set(sources))  # Remove duplicates
        }


def main():
    """Example usage."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import retrieval module
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.retrieval.setup_retrieval import CourseRetriever
    
    print("=" * 60)
    print("Testing Llama 3.2 3B RAG System")
    print("=" * 60)
    
    # Initialize retriever
    print("\n1. Loading retrieval system...")
    db_directory = project_root / "data" / "chroma_db"
    retriever = CourseRetriever(
        collection_name="chalmers_courses",
        persist_directory=str(db_directory)
    )
    
    # Initialize generator
    print("\n2. Loading Llama 3.2 3B model...")
    generator = LlamaRAGGenerator()
    
    # Test query
    print("\n3. Testing RAG pipeline...")
    query = "What courses teach deep learning?"
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    # Retrieve relevant courses
    retrieved = retriever.retrieve(query, top_k=3)
    
    if not retrieved:
        print("No courses retrieved!")
        return
    
    # Format for generator (convert from retriever format)
    context_docs = []
    for r in retrieved:
        context_docs.append({
            'course_code': r['course_code'],
            'course_name': r['course_name'],
            'document': r['document']
        })
    
    # Generate response
    print("\nGenerating response...")
    result = generator.generate_with_sources(query, context_docs)
    
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()

