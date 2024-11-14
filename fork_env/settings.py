import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CLOVERPOOL_API_LIST = [os.environ.get(f"CLOVERPOOL_API_{i}") for i in range(1, 5)]
