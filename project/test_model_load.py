"""Quick test to verify model loading works."""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from src.generation.llama_generator import LlamaRAGGenerator

print("Testing model loading...")
print("Token loaded:", "Yes" if os.getenv("HF_TOKEN") else "No")

try:
    generator = LlamaRAGGenerator(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_alternative_if_gated=False
    )
    print("[OK] Model loaded successfully!")
    
    # Test a simple generation
    print("\nTesting generation...")
    test_context = [{
        'course_code': 'TEST001',
        'course_name': 'Test Course',
        'document': 'This is a test course about machine learning.'
    }]
    
    response = generator.generate(
        "What is this course about?",
        test_context,
        max_new_tokens=50
    )
    print(f"Response: {response}")
    print("\n[OK] Generation test successful!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

