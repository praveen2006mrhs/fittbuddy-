"""Main entry point for the FitBuddy FastAPI application."""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import init_db
from app.routes.health import router as health_router
from app.routes.web import router as web_router
from app.routes.auth import router as auth_router
from app.routes.profile import router as profile_router
from app.routes.plans import router as plans_router
from app.routes.tips import router as tips_router
from app.routes.admin import router as admin_router

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook for startup and shutdown operations."""
    settings = get_settings()
    # Initialize database tables non-destructively
    try:
        init_db()
        print(f"[{settings.app_name}] Database initialized. Server running in {settings.app_env} mode at http://{settings.host}:{settings.port}")
    except Exception as e:
        print(f"[{settings.app_name}] Database initialization error during startup: {e}")
    yield
    print(f"[{settings.app_name}] Shutting down gracefully.")


def create_app() -> FastAPI:
    """Application factory for FitBuddy."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="FitBuddy — AI-Powered Personalized Workout & Fitness Companion",
        version="0.2.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Mount static assets
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Global exception handler to capture and log any unhandled runtime crashes
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.responses import JSONResponse, HTMLResponse

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, StarletteHTTPException):
            return await http_exception_handler(request, exc)
        import traceback
        tb = traceback.format_exc()
        print(f"[FitBuddy Critical Error] {request.method} {request.url.path}: {exc}\n{tb}", flush=True)
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(exc)})
        return HTMLResponse(
            status_code=500,
            content=(
                f"<!DOCTYPE html><html><body style='font-family:sans-serif;padding:2rem;background:#0f172a;color:#f8fafc;'>"
                f"<h2>FitBuddy Server Error</h2>"
                f"<p style='color:#f87171;'>{str(exc)}</p>"
                f"<pre style='background:#1e293b;padding:1rem;border-radius:8px;overflow:auto;'>{tb}</pre>"
                f"</body></html>"
            ),
        )

    # Include route modules
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(profile_router)
    app.include_router(plans_router)
    app.include_router(tips_router)
    app.include_router(admin_router)
    app.include_router(web_router)

    return app


app = create_app()

