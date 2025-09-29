import os
from pathlib import Path
from dotenv import load_dotenv

from src.server import main

# Add src directory to Python path
project_root = Path(__file__).parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


if __name__ == "__main__":
    import asyncio

    print(project_root)
    print(os.getenv("OPENAI_API_KEY"))
    asyncio.run(main())
