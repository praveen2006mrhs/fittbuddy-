"""Security, password hashing (Argon2id), session management, and CSRF protection."""

import hmac
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi import Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User, UserSession, UserRole

settings = get_settings()

# Initialize Argon2 PasswordHasher (OWASP recommended parameters)
_ph = PasswordHasher()

SESSION_COOKIE_NAME = "fitbuddy_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days
CSRF_LIFETIME_SECONDS = 7200  # 2 hours


# ==============================================================================
# Password Hashing
# ==============================================================================

def hash_password(password: str) -> str:
    """Hash password using Argon2id."""
    return _ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against Argon2id hash."""
    try:
        return _ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# ==============================================================================
# Server-Controlled Sessions
# ==============================================================================

def create_user_session(db: Session, user: User) -> UserSession:
    """Create an opaque server-controlled session token in SQLite."""
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_MAX_AGE_SECONDS)
    
    session_record = UserSession(
        session_token=token,
        user_id=user.id,
        expires_at=expires_at,
    )
    db.add(session_record)
    db.commit()
    db.refresh(session_record)
    return session_record


def get_user_by_session_token(db: Session, token: str) -> Optional[User]:
    """Retrieve user by non-expired session token."""
    if not token:
        return None

    now = datetime.now(timezone.utc)
    session_record = (
        db.query(UserSession)
        .filter(UserSession.session_token == token)
        .first()
    )

    if not session_record:
        return None

    # Handle expiration
    # Ensure expires_at is timezone-aware for comparison
    expires_at = session_record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < now:
        db.delete(session_record)
        db.commit()
        return None

    return session_record.user


def delete_user_session(db: Session, token: str) -> None:
    """Revoke and delete server-side session."""
    if not token:
        return
    db.query(UserSession).filter(UserSession.session_token == token).delete()
    db.commit()


# ==============================================================================
# CSRF Protection
# ==============================================================================

def generate_csrf_token(subject: str = "anonymous") -> str:
    """Generate a signed, timed CSRF token bound to the user or anonymous session."""
    salt = secrets.token_hex(8)
    expires_at = int(time.time()) + CSRF_LIFETIME_SECONDS
    payload = f"{salt}.{expires_at}.{subject}"
    signature = hmac.new(
        settings.session_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{signature}"


def verify_csrf_token(token: str, subject: str = "anonymous") -> bool:
    """Verify validity and signature of submitted CSRF token."""
    if not token or not isinstance(token, str):
        return False

    parts = token.split(".")
    if len(parts) != 4:
        return False

    salt, expires_str, token_subject, signature = parts
    try:
        expires_at = int(expires_str)
    except ValueError:
        return False

    if time.time() > expires_at:
        return False

    # Subject must match (e.g. current user ID or "anonymous")
    if token_subject != subject:
        return False

    reconstructed_payload = f"{salt}.{expires_at}.{token_subject}"
    expected_sig = hmac.new(
        settings.session_secret.encode("utf-8"),
        reconstructed_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected_sig)


# ==============================================================================
# FastAPI Dependencies & Session Extraction
# ==============================================================================

def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Extract authenticated user if session cookie is present and valid."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        return None
    return get_user_by_session_token(db, session_token)


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """Enforce authentication. If not authenticated, redirect browser to /login or raise 401."""
    user = get_current_user_optional(request, db)
    if not user:
        accept = request.headers.get("accept", "")
        # If browser GET request, redirect smoothly to /login
        if (
            request.method == "GET"
            and not request.url.path.startswith("/api/")
            and ("text/html" in accept or "*/*" in accept or not accept)
        ):
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                detail="Authentication required",
                headers={"Location": "/login"},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def require_admin_user(
    request: Request,
    user: User = Depends(get_current_user),
) -> User:
    """Enforce administrator role for protected admin routes.
    
    If an ordinary user or non-admin attempts to access, raise 403 Forbidden.
    """
    if user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin authorization required. Access denied.",
        )
    return user
