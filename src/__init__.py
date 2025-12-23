"""
MORL: Multi-Objective Reinforcement Learning Framework
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

__version__ = "0.1.0"
