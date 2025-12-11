"""Timing utilities for performance tracking."""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict


class TimingTracker:
    """Track timing for different operations."""
    
    def __init__(self):
        """Initialize timing tracker."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_timings: Dict[str, float] = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time an operation.
        
        Args:
            operation_name: Name of the operation to time
            
        Example:
            with tracker.time_operation("retrieval"):
                # code to time
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[operation_name].append(elapsed)
            self.current_timings[operation_name] = elapsed
    
    def get_timing(self, operation_name: str) -> Optional[float]:
        """Get the last timing for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Last timing in seconds, or None if not found
        """
        return self.current_timings.get(operation_name)
    
    def get_average_timing(self, operation_name: str) -> Optional[float]:
        """Get average timing for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Average timing in seconds, or None if not found
        """
        timings = self.timings.get(operation_name)
        if not timings:
            return None
        return sum(timings) / len(timings)
    
    def get_total_timing(self, operation_name: str) -> float:
        """Get total timing for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Total timing in seconds (0 if not found)
        """
        return sum(self.timings.get(operation_name, []))
    
    def print_summary(self, title: str = "Timing Summary"):
        """Print a summary of all timings.
        
        Args:
            title: Title for the summary
        """
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        if not self.timings:
            print("No timings recorded.")
            return
        
        for operation_name in sorted(self.timings.keys()):
            timings = self.timings[operation_name]
            avg_time = sum(timings) / len(timings)
            total_time = sum(timings)
            count = len(timings)
            
            print(f"\n{operation_name}:")
            print(f"  Count: {count}")
            print(f"  Average: {avg_time*1000:.2f} ms")
            print(f"  Total: {total_time*1000:.2f} ms")
            if count > 1:
                print(f"  Min: {min(timings)*1000:.2f} ms")
                print(f"  Max: {max(timings)*1000:.2f} ms")
        
        print("=" * 70)
    
    def print_current_timings(self, title: str = "Current Timings"):
        """Print current timings for the last operations.
        
        Args:
            title: Title for the summary
        """
        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)
        
        if not self.current_timings:
            print("No timings recorded.")
            return
        
        for operation_name in sorted(self.current_timings.keys()):
            elapsed = self.current_timings[operation_name]
            print(f"  {operation_name}: {elapsed*1000:.2f} ms")
        
        print("-" * 70)
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.current_timings.clear()


# Global timing tracker instance
_global_tracker = TimingTracker()


def get_tracker() -> TimingTracker:
    """Get the global timing tracker instance.
    
    Returns:
        Global TimingTracker instance
    """
    return _global_tracker


@contextmanager
def time_operation(operation_name: str):
    """Context manager to time an operation using global tracker.
    
    Args:
        operation_name: Name of the operation to time
        
    Example:
        with time_operation("retrieval"):
            # code to time
            pass
    """
    with _global_tracker.time_operation(operation_name):
        yield

