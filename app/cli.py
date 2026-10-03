"""Command-line administrative utility for FitBuddy."""

import sys
import argparse
import getpass
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import User, UserRole
from app.schemas import UserRegisterSchema
from app.services.security import hash_password
from app.config import Settings, get_settings
from app.services.ai import GeminiAIService, AIServiceError


def create_admin(email: str = None, password: str = None) -> bool:
    """Create a new administrator account or promote an existing user to admin."""
    # Ensure tables exist
    init_db()

    print("\n--- FitBuddy Local Admin Setup ---")
    if not email:
        try:
            email = input("Enter admin email address: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return False

    if not password:
        try:
            password = getpass.getpass("Enter admin password (min 8 characters): ")
            confirm = getpass.getpass("Confirm admin password: ")
            if password != confirm:
                print("Error: Passwords do not match.")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return False

    # Validate inputs using Pydantic schema
    try:
        validated = UserRegisterSchema(
            email=email,
            password=password,
            confirm_password=password,
        )
    except ValidationError as e:
        for err in e.errors():
            print(f"Validation Error: {err.get('msg', 'Invalid input')}")
        return False

    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.email == validated.email).first()
        if user:
            print(f"User with email '{validated.email}' already exists. Promoting to 'admin' role...")
            user.role = UserRole.ADMIN.value
            user.password_hash = hash_password(validated.password)
            db.commit()
            print(f"Success: User '{validated.email}' is now an Admin with updated password.")
        else:
            new_admin = User(
                email=validated.email,
                password_hash=hash_password(validated.password),
                role=UserRole.ADMIN.value,
            )
            db.add(new_admin)
            db.commit()
            print(f"Success: Administrator account '{validated.email}' created successfully.")
        return True
    finally:
        db.close()


async def _async_gemini_diagnostics(
    test_request: bool = False,
    model_override: str = None,
    settings: Settings = None,
) -> bool:
    """Run asynchronous diagnostic routine for Gemini integration."""
    import time

    if settings is None:
        settings = get_settings()

    print("\n" + "=" * 70)
    print(" FitBuddy - Google Gemini Integration Diagnostics")
    print("=" * 70)

    # 1. Configuration check (never print the full key!)
    if not settings.is_gemini_configured:
        print("[CONFIG CHECK] Status: NOT CONFIGURED")
        print("  - GEMINI_API_KEY: Empty, missing, or default placeholder.")
        print(f"  - Workout Model : {settings.gemini_workout_model}")
        print(f"  - Tip Model     : {settings.gemini_tip_model}")
        print(f"  - Mock Mode     : {'ENABLED (Explicit Dev Setting)' if settings.gemini_mock_enabled else 'DISABLED'}")
        print("\n[REMAINING CONFIGURATION STEP]")
        print("  To enable live Gemini integration:")
        print("  1. Obtain an API key from Google AI Studio: https://aistudio.google.com/")
        print("  2. Open your local .env file in the project root")
        print("  3. Set: GEMINI_API_KEY=your_actual_key_here")
        print("  4. Re-run: python manage.py gemini-diagnostics")
        print("\n[STATUS] Live integration remains unverified.")
        print("=" * 70 + "\n")
        return False

    print("[CONFIG CHECK] Status: CONFIGURED")
    print(f"  - Key Mask      : {settings.masked_gemini_key}")
    print(f"  - Workout Model : {settings.gemini_workout_model}")
    print(f"  - Tip Model     : {settings.gemini_tip_model}")
    print(f"  - Timeout       : {settings.gemini_timeout_seconds}s")
    print(f"  - Max Retries   : {settings.gemini_max_retries}")
    print(f"  - Mock Mode     : {'ENABLED (Mock will intercept)' if settings.gemini_mock_enabled else 'DISABLED (Live requests active)'}")

    ai_service = GeminiAIService(settings=settings)

    print("\n[API CONNECTIVITY] Querying accessible models via Google GenAI SDK...")
    try:
        models = await ai_service.list_compatible_models()
        print(f"  Successfully retrieved {len(models)} compatible model(s):")
        for m in models:
            in_lim = m.get("input_token_limit") or "N/A"
            out_lim = m.get("output_token_limit") or "N/A"
            print(f"    - {m['name']} ({m['display_name']}) [In: {in_lim}, Out: {out_lim}]")

        target_model = model_override or settings.gemini_workout_model
        model_names = [m["name"] for m in models] + [m["name"].replace("models/", "") for m in models]
        target_clean = target_model.replace("models/", "")
        matched = any(target_clean in m for m in model_names)
        if matched:
            print(f"  [OK] Configured workout model '{target_model}' is available.")
        else:
            print(f"  ! Notice: '{target_model}' was not explicitly in listed models, but may still be valid.")

    except AIServiceError as e:
        print(f"\n[API ERROR] {e.user_message}")
        print(f"  Diagnostic Code: {e.code}")
        print("\n[STATUS] Live model listing failed.")
        print("=" * 70 + "\n")
        return False
    except Exception as e:
        print(f"\n[API ERROR] Unexpected error: {type(e).__name__}: {e}")
        print("\n[STATUS] Live model listing failed.")
        print("=" * 70 + "\n")
        return False

    # 2. Minimal test request if requested
    if test_request:
        target_model = model_override or settings.gemini_workout_model
        print(f"\n[MINIMAL TEST] Sending minimal generation request to '{target_model}'...")
        start_time = time.time()
        try:
            result = await ai_service.generate_minimal_test(model=target_model)
            elapsed = time.time() - start_time
            print(f"  [OK] Success! Response received in {elapsed:.2f}s")
            print(f"  - Model Used    : {result.model_identifier}")
            print(f"  - Output Text   : {result.data}")
            print(f"  - Is Mock       : {result.is_mock}")
            print(f"  - Finish Reason : {result.finish_reason}")
            if result.usage_metadata:
                print(f"  - Token Usage   : {result.usage_metadata}")
            print("\n[STATUS] Live Gemini integration verified successfully!")
            print("=" * 70 + "\n")
            return True
        except AIServiceError as e:
            print(f"\n[TEST FAILED] {e.user_message}")
            print(f"  Diagnostic Code: {e.code}")
            print("\n[STATUS] Live request failed.")
            print("=" * 70 + "\n")
            return False
        except Exception as e:
            print(f"\n[TEST FAILED] Unexpected error: {type(e).__name__}: {e}")
            print("\n[STATUS] Live request failed.")
            print("=" * 70 + "\n")
            return False

    print("\n[STATUS] Configuration and model list verified. (Use --test to execute a live generation request).")
    print("=" * 70 + "\n")
    return True


def run_gemini_diagnostics(
    test_request: bool = False,
    model: str = None,
    settings: Settings = None,
) -> bool:
    """Synchronous runner for gemini diagnostics command."""
    import asyncio
    return asyncio.run(_async_gemini_diagnostics(test_request=test_request, model_override=model, settings=settings))


def main():
    parser = argparse.ArgumentParser(description="FitBuddy Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-admin subparser
    admin_parser = subparsers.add_parser("create-admin", help="Create or promote an administrator account")
    admin_parser.add_argument("--email", type=str, help="Admin email address", default=None)
    admin_parser.add_argument("--password", type=str, help="Admin password", default=None)

    # init-db subparser
    subparsers.add_parser("init-db", help="Initialize database schema non-destructively")

    # gemini-diagnostics subparser
    diag_parser = subparsers.add_parser("gemini-diagnostics", help="Verify Gemini configuration, list models, and test requests")
    diag_parser.add_argument("--test", action="store_true", help="Execute one minimal test generation request")
    diag_parser.add_argument("--model", type=str, default=None, help="Override target model to test")

    args = parser.parse_args()

    if args.command == "create-admin":
        success = create_admin(email=args.email, password=args.password)
        sys.exit(0 if success else 1)
    elif args.command == "init-db":
        init_db()
        print("Database initialized successfully.")
    elif args.command == "gemini-diagnostics":
        success = run_gemini_diagnostics(test_request=args.test, model=args.model)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

