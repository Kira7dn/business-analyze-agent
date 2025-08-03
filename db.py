#!/usr/bin/env python3
"""
Business Analyze Agent - Main Entry Point
"""

import glob
import sys
import os
import json
from vectorDB.utils import default_tokenizer, smart_chunk_markdown_from_text

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


if __name__ == "__main__":
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    knowledge_base_dir = os.path.abspath(knowledge_base_dir)
    md_files = glob.glob(os.path.join(knowledge_base_dir, "*.md"))
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            document_content = f.read()
        chunks = smart_chunk_markdown_from_text(document_content, default_tokenizer)
        # Export to chunks.json in root directory
        with open("chunks.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(chunks))
        print("--------------")
        print(len(chunks))
        print(chunks)
        print("--------------")
