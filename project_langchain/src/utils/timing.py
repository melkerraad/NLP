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
    

