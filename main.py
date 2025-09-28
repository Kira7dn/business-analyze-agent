#!/usr/bin/env python3
"""
Business Analyze Agent - Main Entry Point
"""

import sys
import os

from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables before importing server (ensures Settings picks them up)
load_dotenv()
openai_api_key_loaded = bool(os.getenv("OPENAI_API_KEY"))

from src.server import main


if __name__ == "__main__":
    import asyncio

    print(f"OPENAI_API_KEY loaded: {openai_api_key_loaded}")
    asyncio.run(main())
