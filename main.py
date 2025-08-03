#!/usr/bin/env python3
"""
Business Analyze Agent - Main Entry Point
"""

import sys
import os
from src.server import main

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
