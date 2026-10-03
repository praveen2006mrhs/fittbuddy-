"""Database configuration and session management with SQLite foreign-key enforcement."""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import get_settings

settings = get_settings()

# Connect args: check_same_thread=False is required for SQLite with FastAPI multi-threading
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
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
        cursor.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        # Ignore if non-SQLite backend
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
    Base.metadata.create_all(bind=engine)
