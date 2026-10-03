"""Vercel Serverless Function entry point for FitBuddy FastAPI."""

import sys
from pathlib import Path

# Ensure project root is in Python sys.path so 'app' imports cleanly on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app  # noqa: E402, F401
