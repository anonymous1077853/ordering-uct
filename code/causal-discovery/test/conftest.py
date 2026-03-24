import sys
from pathlib import Path


# Ensure the repository root is on sys.path so `import cdrl` works when running
# tests from the repo checkout (outside the Docker container environment).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

