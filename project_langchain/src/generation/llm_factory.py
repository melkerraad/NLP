"""LLM factory for creating language models with different providers."""

import os
import torch
from typing import Optional, Dict, Any
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMFactory:
    """Factory for creating LLM instances from different providers."""
    
    @staticmethod
    def create_huggingface_llm(
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        top_p: float = 0.9,
        use_compile: bool = True,
        hf_token: Optional[str] = None
    ) -> HuggingFacePipeline:
        """Create HuggingFace LLM pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda" or "cpu")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_compile: Whether to use torch.compile for optimization
            hf_token: HuggingFace token (required for gated models)
            
        Returns:
            HuggingFacePipeline instance
        """
        print(f"Loading HuggingFace model: {model_name}...")
        
        # Get token from environment if not provided
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        if hf_token is None:
            print("[WARNING] No HuggingFace token found.")
            print("   Set HF_TOKEN environment variable or pass hf_token parameter.")
        
        # Auto-detect device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        
        # Load model
        print(f"Loading model (device: {device})...")
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                token=hf_token
            )
            model = model.to("cpu")
        
        # Compile model for faster inference (PyTorch 2.0+)
        if use_compile and device == "cuda" and hasattr(torch, "compile"):
            try:
                print("Compiling model for faster inference...")
                model = torch.compile(model, mode="reduce-overhead")
                print("[OK] Model compiled")
            except Exception as e:
                print(f"[WARNING] Could not compile model: {e}")
        
        model.eval()
        print(f"[OK] Model loaded on {device}")
        
        # Create pipeline
        print("Creating pipeline...")
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0.1 else None,
            top_p=top_p if temperature > 0.1 else None,
            do_sample=temperature > 0.1
        )
        
        # Wrap in LangChain pipeline
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("[OK] LLM pipeline created")
        
        return llm
    
    @staticmethod
    def create(
        provider: str = "huggingface",
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        **kwargs
    ) -> Any:
        """Create LLM from specified provider.
        
        Args:
            provider: LLM provider ("huggingface", "openai", "anthropic", etc.)
            model_name: Model identifier
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider == "huggingface":
            return LLMFactory.create_huggingface_llm(
                model_name=model_name,
                **kwargs
            )
        elif provider == "openai":
            # Placeholder for OpenAI integration
            raise NotImplementedError("OpenAI provider not yet implemented")
        elif provider == "anthropic":
            # Placeholder for Anthropic integration
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {provider}")


def create_llm_from_config(config: Dict[str, Any]) -> Any:
    """Create LLM from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        LLM instance
    """
    model_config = config.get("model", {})
    
    return LLMFactory.create(
        provider="huggingface",  # Currently only HuggingFace supported
        model_name=model_config.get("name", "meta-llama/Llama-3.2-1B-Instruct"),
        device=model_config.get("device", "cuda"),
        max_new_tokens=model_config.get("max_new_tokens", 200),
        temperature=model_config.get("temperature", 0.1),
        top_p=model_config.get("top_p", 0.9),
        use_compile=model_config.get("use_compile", True)
    )

