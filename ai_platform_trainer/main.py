"""
Main entry point for AI Platform Trainer.

This module forwards to the unified launcher for consistency.
"""
import os
import sys

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and use the unified launcher
from unified_launcher import main

if __name__ == "__main__":
    main()