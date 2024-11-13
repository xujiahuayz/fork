import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CLOVERPOOL_API_1 = os.environ.get("CLOVERPOOL_API_1")
CLOVERPOOL_API_2 = os.environ.get("CLOVERPOOL_API_2")
