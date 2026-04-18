"""
tests/conftest.py
Pytest configuration for V3 agentic pipeline tests.

Adds both the project root and v3_agentic to sys.path so tests can use
the same bare-module imports as the production code:
    from pipeline.state import ...
    from agents.utils import ...
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
V3_ROOT = ROOT / "v3_agentic"

# Order matters: v3_agentic first so bare imports (pipeline.*, agents.*) resolve
# to the right modules; project root second for v3_agentic.* full-path imports.
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(ROOT))
