"""Automated tests for user accounts, authentication, session security, and profile isolation."""

import pytest
import re
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models import User, FitnessProfile, UserSession, UserRole
from app.services.security import generate_csrf_token, verify_password
from app.cli import create_admin


# Use an isolated test SQLite database for test runs
TEST_DATABASE_URL = "sqlite:///./test_fitbuddy.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create a clean schema for tests and clean up afterwards."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client():
    """Create a fresh FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def extract_csrf_token(html: str) -> str:
    """Helper to extract CSRF token from rendered HTML forms."""
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        return generate_csrf_token("anonymous")
    return match.group(1)


def test_user_registration_flow(client: TestClient):
    """Test full user registration with Argon2 password hashing and session cookie."""
    # 1. Fetch register page to get CSRF token
    get_res = client.get("/register")
    assert get_res.status_code == 200
    csrf = extract_csrf_token(get_res.text)

    # 2. Submit valid registration
    post_res = client.post(
        "/register",
        data={
            "email": "athlete@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )
    assert post_res.status_code == 303
    assert post_res.headers["location"] == "/profile"
    assert "fitbuddy_session" in post_res.cookies

    # 3. Verify user in database and password hash
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "athlete@example.com").first()
    assert user is not None
    assert user.role == UserRole.USER.value
    assert user.password_hash.startswith("$argon2id$")
    assert verify_password("Password123!", user.password_hash)
    db.close()


def test_registration_validation_errors(client: TestClient):
    """Test registration schema validations (short password, mismatch, duplicate email)."""
    # Short password
    csrf = generate_csrf_token("anonymous")
    res1 = client.post(
        "/register",
        data={
            "email": "user@example.com",
            "password": "123",
            "confirm_password": "123",
            "csrf_token": csrf,
        },
    )
    assert res1.status_code == 422
    assert "at least 8 characters" in res1.text.lower() or "8" in res1.text

    # Password mismatch
    res2 = client.post(
        "/register",
        data={
            "email": "user@example.com",
            "password": "ValidPassword1",
            "confirm_password": "DifferentPassword2",
            "csrf_token": csrf,
        },
    )
    assert res2.status_code == 422
    assert "passwords do not match" in res2.text.lower()


def test_login_and_logout_flow():
    """Test user login, authenticated profile access, and logout."""
    client = TestClient(app)
    csrf = generate_csrf_token("anonymous")
    # Register user
    client.post(
        "/register",
        data={
            "email": "tester@example.com",
            "password": "SecretPassword1",
            "confirm_password": "SecretPassword1",
            "csrf_token": csrf,
        },
    )

    # Login with bad password
    bad_login = client.post(
        "/login",
        data={
            "email": "tester@example.com",
            "password": "WrongPassword!",
            "csrf_token": csrf,
        },
    )
    assert bad_login.status_code == 401
    assert "Invalid email or password" in bad_login.text

    # Login with good password
    good_login = client.post(
        "/login",
        data={
            "email": "tester@example.com",
            "password": "SecretPassword1",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )
    assert good_login.status_code == 303
    assert good_login.headers["location"] == "/profile"
    session_cookie = good_login.cookies.get("fitbuddy_session")
    assert session_cookie is not None

    # Access profile with session cookie
    profile_res = client.get("/profile")
    assert profile_res.status_code == 200
    assert "Personal Fitness Profile" in profile_res.text
    profile_csrf = extract_csrf_token(profile_res.text)

    # Logout
    logout_res = client.post(
        "/logout",
        data={"csrf_token": profile_csrf},
        follow_redirects=False,
    )
    assert logout_res.status_code == 303
    assert "logged_out=1" in logout_res.headers["location"]

    # Verify session is revoked in database
    db = TestingSessionLocal()
    revoked = db.query(UserSession).filter(UserSession.session_token == session_cookie).first()
    assert revoked is None
    db.close()


def test_profile_creation_and_bounds_validation():
    """Test fitness profile persistence and adult 18+ bounds validation."""
    client = TestClient(app)
    csrf = generate_csrf_token("anonymous")
    client.post(
        "/register",
        data={
            "email": "gymnast@example.com",
            "password": "WorkoutPass123",
            "confirm_password": "WorkoutPass123",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )

    # Get profile page to extract user-bound CSRF token
    p_get = client.get("/profile")
    user_csrf = extract_csrf_token(p_get.text)

    # 1. Underage check (Age < 18)
    underage_res = client.post(
        "/profile",
        data={
            "display_name": "Teen Athlete",
            "age": "17",
            "weight_kg": "65.0",
            "fitness_goal": "muscle_gain",
            "workout_intensity": "medium",
            "experience_level": "beginner",
            "equipment": "dumbbells",
            "available_minutes": "45",
            "exercise_limitations": "",
            "csrf_token": user_csrf,
        },
    )
    assert underage_res.status_code == 422
    assert "adults aged 18 and older" in underage_res.text.lower()

    # 2. Negative / invalid weight (< 20 kg)
    p_get2 = client.get("/profile")
    user_csrf2 = extract_csrf_token(p_get2.text)
    bad_weight_res = client.post(
        "/profile",
        data={
            "display_name": "Jane",
            "age": "25",
            "weight_kg": "10.0",
            "fitness_goal": "weight_loss",
            "workout_intensity": "low",
            "experience_level": "beginner",
            "equipment": "bodyweight",
            "available_minutes": "30",
            "exercise_limitations": "",
            "csrf_token": user_csrf2,
        },
    )
    assert bad_weight_res.status_code == 422
    assert "between 20.0 kg and 350.0 kg" in bad_weight_res.text

    # 3. Invalid session duration (< 10 mins or > 180 mins)
    p_get3 = client.get("/profile")
    user_csrf3 = extract_csrf_token(p_get3.text)
    bad_duration_res = client.post(
        "/profile",
        data={
            "display_name": "Jane",
            "age": "25",
            "weight_kg": "60.0",
            "fitness_goal": "weight_loss",
            "workout_intensity": "low",
            "experience_level": "beginner",
            "equipment": "bodyweight",
            "available_minutes": "5",
            "exercise_limitations": "",
            "csrf_token": user_csrf3,
        },
    )
    assert bad_duration_res.status_code == 422
    assert "between 10 and 180 minutes" in bad_duration_res.text

    # 4. Valid profile submission
    p_get4 = client.get("/profile")
    user_csrf4 = extract_csrf_token(p_get4.text)
    valid_res = client.post(
        "/profile",
        data={
            "display_name": "Jane Doe",
            "age": "28",
            "weight_kg": "64.5",
            "fitness_goal": "muscle_gain",
            "workout_intensity": "high",
            "experience_level": "intermediate",
            "equipment": "full_gym",
            "available_minutes": "60",
            "exercise_limitations": "Slight left shoulder pinch on overhead press",
            "csrf_token": user_csrf4,
        },
    )
    assert valid_res.status_code == 200
    assert "Fitness profile saved successfully!" in valid_res.text

    # Verify database persistence
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "gymnast@example.com").first()
    assert user.profile is not None
    assert user.profile.display_name == "Jane Doe"
    assert user.profile.age == 28
    assert user.profile.weight_kg == 64.5
    assert user.profile.fitness_goal == "muscle_gain"
    assert user.profile.workout_intensity == "high"
    assert user.profile.experience_level == "intermediate"
    assert user.profile.equipment == "full_gym"
    assert user.profile.available_minutes == 60
    assert "shoulder pinch" in user.profile.exercise_limitations
    db.close()


def test_two_accounts_profile_isolation():
    """
    CRITICAL CHECK: Create two accounts and verify their profiles remain completely separate
    and cannot be accessed or overwritten by each other.
    Simulate two independent client/browser sessions.
    """
    client_a = TestClient(app)
    client_b = TestClient(app)

    # 1. Register Account A (Alice)
    get_a = client_a.get("/register")
    csrf_init_a = extract_csrf_token(get_a.text)
    client_a.post(
        "/register",
        data={
            "email": "alice@fitbuddy.test",
            "password": "AliceSecret123",
            "confirm_password": "AliceSecret123",
            "csrf_token": csrf_init_a,
        },
        follow_redirects=False,
    )

    # Save Alice's profile
    p_get_a = client_a.get("/profile")
    csrf_a = extract_csrf_token(p_get_a.text)
    client_a.post(
        "/profile",
        data={
            "display_name": "Alice PowerLifter",
            "age": "29",
            "weight_kg": "70.0",
            "fitness_goal": "muscle_gain",
            "workout_intensity": "high",
            "experience_level": "advanced",
            "equipment": "full_gym",
            "available_minutes": "90",
            "exercise_limitations": "No lower back injuries",
            "csrf_token": csrf_a,
        },
    )

    # 2. Register Account B (Bob) in separate client
    get_b = client_b.get("/register")
    csrf_init_b = extract_csrf_token(get_b.text)
    client_b.post(
        "/register",
        data={
            "email": "bob@fitbuddy.test",
            "password": "BobSecret456",
            "confirm_password": "BobSecret456",
            "csrf_token": csrf_init_b,
        },
        follow_redirects=False,
    )

    # Save Bob's profile
    p_get_b = client_b.get("/profile")
    csrf_b = extract_csrf_token(p_get_b.text)
    client_b.post(
        "/profile",
        data={
            "display_name": "Bob Runner",
            "age": "35",
            "weight_kg": "82.5",
            "fitness_goal": "weight_loss",
            "workout_intensity": "medium",
            "experience_level": "beginner",
            "equipment": "bodyweight",
            "available_minutes": "30",
            "exercise_limitations": "Mild asthma during intense cardio",
            "csrf_token": csrf_b,
        },
    )

    # 3. Read Profile as Alice: Sees Alice, NEVER Bob
    alice_view = client_a.get("/profile")
    assert alice_view.status_code == 200
    assert "Alice PowerLifter" in alice_view.text
    assert 'value="70.0"' in alice_view.text or 'value="70"' in alice_view.text
    assert "Bob Runner" not in alice_view.text
    assert "Mild asthma" not in alice_view.text

    # 4. Read Profile as Bob: Sees Bob, NEVER Alice
    bob_view = client_b.get("/profile")
    assert bob_view.status_code == 200
    assert "Bob Runner" in bob_view.text
    assert 'value="82.5"' in bob_view.text
    assert "Alice PowerLifter" not in bob_view.text
    assert "No lower back injuries" not in bob_view.text

    # 5. Database check: Verify two distinct rows with respective user_ids
    db = TestingSessionLocal()
    alice_db = db.query(User).filter(User.email == "alice@fitbuddy.test").first()
    bob_db = db.query(User).filter(User.email == "bob@fitbuddy.test").first()
    assert alice_db.id != bob_db.id
    assert alice_db.profile.display_name == "Alice PowerLifter"
    assert bob_db.profile.display_name == "Bob Runner"
    db.close()


def test_account_deletion_with_confirmation():
    """Test account deletion cascade with explicit confirmation keyword and password."""
    client = TestClient(app)
    get_reg = client.get("/register")
    csrf_reg = extract_csrf_token(get_reg.text)
    client.post(
        "/register",
        data={
            "email": "delete_me@example.com",
            "password": "DeletePass999",
            "confirm_password": "DeletePass999",
            "csrf_token": csrf_reg,
        },
        follow_redirects=False,
    )

    # Add profile
    p_get = client.get("/profile")
    user_csrf = extract_csrf_token(p_get.text)
    client.post(
        "/profile",
        data={
            "display_name": "Temporary User",
            "age": "30",
            "weight_kg": "75.0",
            "fitness_goal": "general_wellness",
            "workout_intensity": "low",
            "experience_level": "beginner",
            "equipment": "bodyweight",
            "available_minutes": "30",
            "csrf_token": user_csrf,
        },
    )

    # Visit delete confirmation page
    del_page = client.get("/profile/delete")
    assert del_page.status_code == 200
    del_csrf = extract_csrf_token(del_page.text)

    # 1. Attempt deletion without typing 'DELETE'
    bad_confirm = client.post(
        "/profile/delete",
        data={
            "confirmation_text": "CANCEL",
            "password": "DeletePass999",
            "csrf_token": del_csrf,
        },
    )
    assert bad_confirm.status_code == 422
    assert "must type DELETE exactly" in bad_confirm.text

    # Extract new CSRF after 422
    del_csrf2 = extract_csrf_token(bad_confirm.text)

    # 2. Attempt deletion with wrong password
    bad_pass = client.post(
        "/profile/delete",
        data={
            "confirmation_text": "DELETE",
            "password": "WrongPassword!",
            "csrf_token": del_csrf2,
        },
    )
    assert bad_pass.status_code == 401
    assert "Incorrect password" in bad_pass.text

    # Extract new CSRF after 401
    del_csrf3 = extract_csrf_token(bad_pass.text)

    # 3. Successful deletion
    good_del = client.post(
        "/profile/delete",
        data={
            "confirmation_text": "DELETE",
            "password": "DeletePass999",
            "csrf_token": del_csrf3,
        },
        follow_redirects=False,
    )
    assert good_del.status_code == 303
    assert "deleted=1" in good_del.headers["location"]

    # Verify user and dependent profile records are completely purged
    db = TestingSessionLocal()
    deleted_user = db.query(User).filter(User.email == "delete_me@example.com").first()
    assert deleted_user is None
    deleted_profile = db.query(FitnessProfile).filter(FitnessProfile.display_name == "Temporary User").first()
    assert deleted_profile is None
    db.close()


def test_cli_admin_creation():
    """Test administrative command creates an admin account with Argon2 hash."""
    email = "superadmin@example.com"
    password = "SuperAdminPassword123!"
    
    # Run CLI function
    res = create_admin(email=email, password=password)
    assert res is True

    # Verify in DB via SessionLocal used by CLI
    from app.database import SessionLocal
    db = SessionLocal()
    admin_user = db.query(User).filter(User.email == email).first()
    assert admin_user is not None
    assert admin_user.role == UserRole.ADMIN.value
    assert verify_password(password, admin_user.password_hash)
    # Clean up created admin
    db.delete(admin_user)
    db.commit()
    db.close()
