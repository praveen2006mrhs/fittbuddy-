"""Database configuration and session management with SQLite foreign-key enforcement."""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import get_settings

settings = get_settings()

db_url = settings.database_url

# Vercel / AWS Lambda Serverless Compatibility:
# On Vercel, the local filesystem is read-only except for /tmp.
# If database_url is the default relative SQLite URL, redirect it to /tmp so table creation succeeds.
is_serverless = bool(
    os.environ.get("VERCEL")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or not os.access(".", os.W_OK)
)
if is_serverless and db_url.startswith("sqlite") and not db_url.startswith("sqlite:////tmp") and not db_url.startswith("sqlite:///:memory:"):
    db_url = "sqlite:////tmp/fitbuddy.db"


# Connect args: check_same_thread=False is required for SQLite with FastAPI multi-threading
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    connect_args=connect_args,
    echo=False,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enforce SQLite foreign key constraints and enable WAL mode for performance."""
    # Only execute SQLite pragmas if the underlying connection is sqlite3
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON;")
        if is_serverless:
            cursor.execute("PRAGMA journal_mode=MEMORY;")
        else:
            cursor.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        # Ignore if non-SQLite backend or unsupported pragma
        pass
    finally:
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency yielding a scoped database session with lifecycle cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Non-destructive table initialization creating tables if they do not exist."""
    # Import models here so that Base.metadata has all table definitions registered
    import app.models  # noqa: F401
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[FitBuddy Database] Warning: Could not create tables: {e}")
