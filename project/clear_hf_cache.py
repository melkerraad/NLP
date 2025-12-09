"""Clear Hugging Face cache to free up disk space."""

from huggingface_hub import scan_cache_dir
from pathlib import Path
import shutil
import os

def clear_hf_cache():
    """Clear Hugging Face cache."""
    print("Scanning Hugging Face cache...")
    
    try:
        # Scan cache
        cache_info = scan_cache_dir()
        
        print(f"\nTotal cache size: {cache_info.size_on_disk_str}")
        
        # Try to get revisions
        try:
            revisions = list(cache_info.revisions)
            print(f"Number of cached models: {len(revisions)}")
            
            if revisions:
                print("\nCached models:")
                for rev in revisions[:10]:
                    print(f"  - {rev}")
                if len(revisions) > 10:
                    print(f"  ... and {len(revisions) - 10} more")
        except:
            print("(Could not list individual models)")
        
        # Ask what to delete
        print("\nOptions:")
        print("1. Delete all cache (frees most space)")
        print("2. Cancel")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            print("\nDeleting all cache...")
            try:
                # Try new API
                delete_strategy = cache_info.delete_revisions("*")
                delete_strategy.execute()
            except:
                # Fallback: manual deletion
                cache_path = Path.home() / ".cache" / "huggingface" / "hub"
                if cache_path.exists():
                    print(f"Manually deleting: {cache_path}")
                    shutil.rmtree(cache_path)
                    print("✅ Cache cleared!")
                else:
                    print("Cache path not found")
            print("✅ Cache cleared!")
        else:
            print("Cancelled.")
            return
            
    except Exception as e:
        print(f"Error scanning cache: {e}")
        print("\nManual deletion option:")
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"Cache location: {cache_path}")
        
        if cache_path.exists():
            size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024**3)
            print(f"Cache size: {size:.2f} GB")
            
            choice = input("\nDelete cache folder? (y/n): ").strip().lower()
            if choice == 'y':
                try:
                    shutil.rmtree(cache_path)
                    print("✅ Cache deleted!")
                except Exception as e2:
                    print(f"❌ Error deleting: {e2}")
                    print("\nTry deleting manually:")
                    print(f"  {cache_path}")
        else:
            print("Cache folder not found (may already be empty)")

if __name__ == "__main__":
    try:
        clear_hf_cache()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative: Manually delete cache folder:")
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"  {cache_path}")
        print("\nOr on Windows:")
        print(f"  C:\\Users\\<your_username>\\.cache\\huggingface\\hub")

