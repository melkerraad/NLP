"""Quick setup verification script for Llama 3.2 3B."""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    missing = []
    
    try:
        import torch
        print(f"  ✅ torch ({torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️  CUDA not available (will use CPU - slower)")
    except ImportError:
        missing.append("torch")
        print("  ❌ torch not installed")
    
    try:
        import transformers
        print(f"  ✅ transformers ({transformers.__version__})")
    except ImportError:
        missing.append("transformers")
        print("  ❌ transformers not installed")
    
    try:
        import accelerate
        print(f"  ✅ accelerate ({accelerate.__version__})")
    except ImportError:
        missing.append("accelerate")
        print("  ❌ accelerate not installed")
    
    return missing

def check_hf_token():
    """Check if Hugging Face token is set."""
    import os
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if token:
        print(f"  ✅ HF_TOKEN found (length: {len(token)})")
        return True
    else:
        print("  ❌ HF_TOKEN not found")
        print("     Set it with: $env:HF_TOKEN='your_token' (PowerShell)")
        print("     Or: export HF_TOKEN='your_token' (Linux/Mac)")
        print("     Get token at: https://huggingface.co/settings/tokens")
        return False

def check_data():
    """Check if retrieval data exists."""
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "chroma_db"
    courses_path = project_root / "data" / "processed" / "courses_clean.json"
    
    print("\nChecking data...")
    
    if db_path.exists():
        print(f"  ✅ Vector database exists: {db_path}")
    else:
        print(f"  ⚠️  Vector database not found: {db_path}")
        print("     Run: python -m src.retrieval.setup_retrieval")
    
    if courses_path.exists():
        print(f"  ✅ Courses data exists: {courses_path}")
    else:
        print(f"  ⚠️  Courses data not found: {courses_path}")
        print("     Run preprocessing first")

def main():
    """Run all checks."""
    print("=" * 60)
    print("Llama 3.2 3B Setup Verification")
    print("=" * 60)
    
    # Check dependencies
    print("\n[1/3] Dependencies")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return
    
    # Check HF token
    print("\n[2/3] Hugging Face Token")
    has_token = check_hf_token()
    
    # Check data
    check_data()
    
    # Summary
    print("\n" + "=" * 60)
    if missing:
        print("❌ Setup incomplete - install missing packages")
    elif not has_token:
        print("⚠️  Setup incomplete - set HF_TOKEN")
    else:
        print("✅ Setup looks good!")
        print("\nNext steps:")
        print("  1. Test retrieval: python test_retrieval.py")
        print("  2. Test full RAG: python test_rag_full.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

