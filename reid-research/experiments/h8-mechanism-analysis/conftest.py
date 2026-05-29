"""Add this dir to sys.path so test files can `import retrieval`, `import cka`, etc.

The parent dirs (`reid-research`, `h8-mechanism-analysis`) contain dashes and so
cannot be imported as Python packages — flat imports are the workaround.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
