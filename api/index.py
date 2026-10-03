"""Vercel Serverless Function entry point for FitBuddy FastAPI."""

import os
import sys
from pathlib import Path

# Mark Vercel serverless environment
os.environ.setdefault("VERCEL", "1")

# Ensure project root is in Python sys.path so 'app' imports cleanly on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.main import app
    handler = app
except Exception as e:
    import traceback
    crash_traceback = traceback.format_exc()
    print(f"[FitBuddy Cold Start Crash]: {crash_traceback}", flush=True)

    async def fallback_app(scope, receive, send):
        if scope["type"] == "http":
            body = (
                f"<!DOCTYPE html><html><body style='font-family:sans-serif;padding:2rem;background:#0f172a;color:#f8fafc;'>"
                f"<h2 style='color:#ef4444;'>FitBuddy Serverless Cold Start Error</h2>"
                f"<p style='color:#f87171;'>{str(e)}</p>"
                f"<pre style='background:#1e293b;padding:1.5rem;border-radius:8px;overflow:auto;color:#e2e8f0;font-size:14px;line-height:1.5;'>{crash_traceback}</pre>"
                f"</body></html>"
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })

    app = fallback_app
    handler = fallback_app

