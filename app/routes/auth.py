"""Authentication routes for registration, login, and logout."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User, UserRole
from app.schemas import UserRegisterSchema, UserLoginSchema
from app.services.security import (
    hash_password,
    verify_password,
    create_user_session,
    delete_user_session,
    generate_csrf_token,
    verify_csrf_token,
    get_current_user_optional,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Authentication"])
settings = get_settings()


@router.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Render registration page if not already authenticated."""
    if user:
        return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)

    csrf_token = generate_csrf_token(subject="anonymous")
    return templates.TemplateResponse(
        request=request,
        name="auth/register.html",
        context={
            "app_name": settings.app_name,
            "csrf_token": csrf_token,
            "errors": [],
            "email_value": "",
        },
    )


@router.post("/register", response_class=HTMLResponse)
async def handle_register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    """Handle new user registration with password hashing and session creation."""
    # Verify CSRF
    if not verify_csrf_token(csrf_token, subject="anonymous"):
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/register.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": ["Security check failed (invalid or expired CSRF token). Please try again."],
                "email_value": email,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Validate schema
    errors = []
    try:
        validated = UserRegisterSchema(
            email=email,
            password=password,
            confirm_password=confirm_password,
        )
    except ValidationError as exc:
        for err in exc.errors():
            errors.append(err["msg"])
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/register.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": errors,
                "email_value": email,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # Check if user already exists
    existing = db.query(User).filter(User.email == validated.email).first()
    if existing:
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/register.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": ["An account with this email address already exists. Please log in."],
                "email_value": validated.email,
            },
            status_code=status.HTTP_409_CONFLICT,
        )

    # Create user
    new_user = User(
        email=validated.email,
        password_hash=hash_password(validated.password),
        role=UserRole.USER.value,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create server-controlled session
    session_record = create_user_session(db, new_user)

    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_record.session_token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=(settings.app_env == "production"),
        path="/",
    )
    return response


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Render login page if not already authenticated."""
    if user:
        return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)

    csrf_token = generate_csrf_token(subject="anonymous")
    return templates.TemplateResponse(
        request=request,
        name="auth/login.html",
        context={
            "app_name": settings.app_name,
            "csrf_token": csrf_token,
            "errors": [],
            "email_value": "",
        },
    )


@router.post("/login", response_class=HTMLResponse)
async def handle_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    """Authenticate user credentials and establish server session."""
    if not verify_csrf_token(csrf_token, subject="anonymous"):
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/login.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": ["Security check failed (invalid or expired CSRF token). Please try again."],
                "email_value": email,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    errors = []
    try:
        validated = UserLoginSchema(email=email, password=password)
    except ValidationError as exc:
        for err in exc.errors():
            errors.append(err["msg"])
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/login.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": errors,
                "email_value": email,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    user = db.query(User).filter(User.email == validated.email).first()
    if not user or not verify_password(validated.password, user.password_hash):
        new_csrf = generate_csrf_token(subject="anonymous")
        return templates.TemplateResponse(
            request=request,
            name="auth/login.html",
            context={
                "app_name": settings.app_name,
                "csrf_token": new_csrf,
                "errors": ["Invalid email or password. Please check your credentials."],
                "email_value": validated.email,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # Establish server-controlled session
    session_record = create_user_session(db, user)

    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_record.session_token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=(settings.app_env == "production"),
        path="/",
    )
    return response


@router.post("/logout")
async def handle_logout(
    request: Request,
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Invalidate session and remove authentication cookie."""
    subject = str(user.id) if user else "anonymous"
    if not verify_csrf_token(csrf_token, subject=subject):
        # Even if CSRF token is invalid, allow safe logout redirect
        pass

    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        delete_user_session(db, session_token)

    response = RedirectResponse(url="/login?logged_out=1", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )
    return response
