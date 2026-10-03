# FitBuddy ⚡
> Intelligent, Local-First Personalized Workout & Fitness Companion powered by Google Gemini and FastAPI.

FitBuddy is an intelligent, single-process web application designed for coursework demonstrations and portfolio reviews. It generates structured, personalized 7-day workout plans using Google Gemini (`google-genai` SDK), collects user feedback to produce versioned plan revisions, delivers tailored nutrition and recovery tips, persists all state across restarts in SQLite (WAL mode), and includes a role-based, protected administrative dashboard.

---

## Table of Contents

1. [Technology Stack & Architecture](#technology-stack--architecture)
2. [Core Features](#core-features)
3. [Controlled Parameters & Validation Bounds](#controlled-parameters--validation-bounds)
4. [Project Directory Layout](#project-directory-layout)
5. [Complete API Endpoint Table](#complete-api-endpoint-table)
6. [Beginner-Friendly Windows PowerShell Setup](#beginner-friendly-windows-powershell-setup)
7. [Gemini Model Selection & API Key Configuration](#gemini-model-selection--api-key-configuration)
8. [Automated Testing & Smoke Diagnostics](#automated-testing--smoke-diagnostics)
9. [Three-Minute Demonstration Script](#three-minute-demonstration-script)
10. [Concise Architectural Explanation](#concise-architectural-explanation)
11. [Honest Limitations & Future Improvements](#honest-limitations--future-improvements)
12. [Final Verification & Status Table](#final-verification--status-table)
13. [Troubleshooting Guide](#troubleshooting-guide)

---

## Technology Stack & Architecture

- **Backend**: Python 3.11+ (verified on Python 3.14), [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Database Engine**: [SQLAlchemy 2.0](https://www.sqlalchemy.org/) with local SQLite (`fitbuddy.db`), Write-Ahead Logging (`PRAGMA journal_mode=WAL`), and foreign key enforcement (`PRAGMA foreign_keys=ON`)
- **Security & Authentication**:
  - [Argon2id](https://pypi.org/project/argon2-cffi/) password hashing (OWASP recommended parameters)
  - Server-side opaque session records stored in SQLite with secure cookies (`HttpOnly`, `SameSite=Lax`)
  - Cryptographic HMAC-SHA256 CSRF protection bound to user sessions with 2-hour sliding expiry
- **Schema Validation**: [Pydantic v2](https://docs.pydantic.dev/) for strict bounds validation and structured AI outputs
- **Frontend / Templating**: [Jinja2](https://palletsprojects.com/p/jinja/) with automatic HTML-escaping, Semantic HTML5, Vanilla CSS (athletic dark glassmorphism design system), and lightweight client JavaScript
- **AI Integration**: Official Google GenAI SDK (`google-genai` v2.0+), structured schema generation, bounded repair loop, and separate deterministic mock simulation for local testing without spending API quota
- **Test Automation**: [pytest](https://docs.pytest.org/), [httpx](https://www.python-httpx.org/), [Starlette TestClient](https://www.starlette.io/testclient/)

---

## Core Features

1. **Authentication & Adult Scope**:
   - Secure registration, login, and logout.
   - Dedicated adult user scope: strictly enforces age 18 to 120.
   - Isolated profiles: users cannot inspect, edit, or delete another user's profile.
   - Self-service account deletion with confirmation keyword (`DELETE`) and password verification, cascading cleanly across all owned records.
2. **Personalized 7-Day Workout Generator**:
   - Tailored to age, weight, goal, intensity, experience, equipment, available minutes, and exercise limitations.
   - Enforces structured JSON schema: Title, Goal, Summary, General Guidance, and exactly 7 numbered days in chronological order.
   - Validates warm-up, exercises (sets, reps/duration, rest, instructions), cooldown, and recovery notes.
   - Safety rules: beginner cannot receive advanced routines; rest days supported without forcing sets/reps; never defaults to 7 hard training days.
   - Non-blocking: database transactions are never held open during slow Gemini network calls.
3. **Feedback-Based Plan Revisions**:
   - Allows users to provide natural language feedback (e.g. "more cardio", "more rest for sore knees", "only 20 minutes").
   - Treats feedback as untrusted user input with prompt injection delimiters.
   - Produces a complete revised 7-day schedule with a concise change summary.
   - Version lineage: saves as a new version (e.g. v2) linked via `parent_plan_id` while preserving v1.
   - Does not alter the user's base profile.
   - Bounded repair loop: repairs output if schema validation fails on first attempt.
4. **Nutrition & Recovery Tips**:
   - Generates actionable, concise tips for goals (`weight_loss`, `muscle_gain`, `general_wellness`) and categories (`nutrition`, `recovery`).
   - General wellness guidance avoiding medical claims, restrictive diets, or supplement dosing.
   - Displays latest tip and a scrolling history of previous tips.
   - Rate limiting and debounce protection against accidental double submissions.
5. **Protected Admin Dashboard**:
   - Accessible only to accounts with `role="admin"`.
   - Admin status can only be granted via local CLI (`python manage.py create-admin`).
   - Ordinary authenticated users typing `/admin` receive `403 Forbidden`.
   - Live telemetry: actual counts of users, plans, revisions, and tips from SQLite.
   - Paginated user directory with search filtering by email or display name.
   - Deep inspection of user profiles, saved plan history, and tip history.
   - Deep inspection of plan versions, triggering feedback, and child revisions.
   - Transparent indicators badging `Mock Simulation Mode` vs `Live AI Generation`.
   - Strictly read-only for MVP: never exposes password hashes, secrets, or API keys.

---

## Controlled Parameters & Validation Bounds

### 1. Controlled Enums
- **Goals**: `weight_loss`, `muscle_gain`, `general_wellness`
- **Intensity**: `low`, `medium`, `high`
- **Experience Levels**: `beginner`, `intermediate`, `advanced`
- **Equipment**: `bodyweight`, `dumbbells`, `full_gym`
- **Tip Categories**: `nutrition`, `recovery`
- **User Roles**: `user`, `admin`

### 2. Validation Bounds
- **Age**: `18` to `120` years (strictly enforced adults-only scope)
- **Weight**: `20.0` kg to `350.0` kg (finite positive float)
- **Session Duration**: `10` to `180` minutes
- **Display Name**: `2` to `60` characters
- **Password**: Minimum `8` characters, maximum `128` characters (Argon2id)
- **Feedback Text**: `5` to `1,000` characters
- **Exercise Limitations**: Optional string up to `500` characters

---

## Project Directory Layout

```text
.
├── app/
│   ├── __init__.py           # Application package
│   ├── config.py             # Pydantic Settings & environment configuration
│   ├── database.py           # SQLite connection, WAL mode, foreign keys & get_db
│   ├── models.py             # SQLAlchemy 2.0 declarative models
│   ├── schemas.py            # Pydantic validation schemas & bounds
│   ├── cli.py                # Administrative CLI commands & diagnostics
│   ├── main.py               # FastAPI factory, lifespan, and router mounting
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py         # GET /health health-check endpoint
│   │   ├── auth.py           # Authentication routes (/register, /login, /logout)
│   │   ├── profile.py        # Profile management (/profile, /profile/delete)
│   │   ├── plans.py          # 7-day workout generator & revisions (/plans)
│   │   ├── tips.py           # Nutrition & recovery tips (/tips)
│   │   ├── admin.py          # Protected admin dashboard (/admin)
│   │   └── web.py            # Home landing page (/)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai.py             # Gemini GenAI service, structured schemas, bounded repair
│   │   └── security.py       # Argon2id hashing, session tokens, CSRF & require_admin_user
│   ├── static/
│   │   ├── css/style.css     # Athletic dark-mode glassmorphic styling
│   │   └── js/app.js         # Client-side ping and interactivity
│   └── templates/
│       ├── base.html         # Base Jinja2 layout with navigation and alerts
│       ├── index.html        # Landing page with feature cards
│       ├── auth/             # Login and registration templates
│       ├── profile/          # Profile form and deletion dialog templates
│       ├── plans/            # Plan list, generator, and detail inspection templates
│       ├── tips/             # Nutrition and recovery tip generation templates
│       └── admin/            # Admin dashboard, user detail, and plan detail templates
├── tests/
│   ├── __init__.py
│   ├── test_health.py        # Health endpoint & static assets
│   ├── test_ai_service.py    # Gemini service, mock fallback & error handling
│   ├── test_auth_and_profile.py # Registration, login, bounds, deletion & isolation
│   ├── test_workout_plans.py # 7-day schema, bounded repair, revisions & duplicate clicks
│   ├── test_tips.py          # Tips generation, categories, debounce rate-limiting
│   ├── test_admin.py         # Admin RBAC, direct URL 403 denial, telemetry & security
│   ├── test_restart_persistence.py # Verifies SQLite persistence across restarts
│   └── test_cli_diagnostics.py # CLI setup and model listing diagnostics
├── .env.example              # Template configuration with placeholders
├── .gitignore                # Excludes secrets, venv, and sqlite databases
├── requirements.txt          # Python dependencies
├── run.py                    # Local runner binding to 127.0.0.1:8000
├── manage.py                 # CLI shortcut for management tasks
└── README.md                 # Complete project documentation & setup
```

---

## Complete API Endpoint Table

| Method | Path | Auth / Role | Description |
|---|---|---|---|
| `GET` | `/` | Public | FitBuddy landing page with feature cards and live health badge |
| `GET` | `/health` | Public | System health check (database, environment, Gemini config status) |
| `GET` | `/register` | Public | Renders new user registration page |
| `POST` | `/register` | Public | Creates new user with Argon2id hash and server session |
| `GET` | `/login` | Public | Renders login page |
| `POST` | `/login` | Public | Validates credentials and sets `fitbuddy_session` cookie |
| `POST` | `/logout` | Authenticated | Revokes server session and clears cookie |
| `GET` | `/profile` | Authenticated | Renders user fitness profile form with saved values |
| `POST` | `/profile` | Authenticated | Validates and persists user fitness profile parameters |
| `GET` | `/profile/delete` | Authenticated | Renders account deletion confirmation page |
| `POST` | `/profile/delete` | Authenticated | Cascade deletes user account upon password and "DELETE" confirmation |
| `GET` | `/plans` | Authenticated | Lists all saved workout plans for current user |
| `POST` | `/plans/generate` | Authenticated | Generates personalized 7-day plan via Gemini (HTML flow) |
| `POST` | `/api/plans/generate` | Authenticated | Generates personalized 7-day plan (JSON API flow) |
| `GET` | `/plans/{id}` | Owner / Admin | Displays 7-day plan schedule, version history, and revision form |
| `GET` | `/api/plans` | Authenticated | Lists user's saved plans in JSON |
| `GET` | `/api/plans/{id}` | Owner / Admin | Returns structured 7-day plan JSON |
| `POST` | `/plans/{id}/revise` | Owner / Admin | Submits natural-language feedback and creates revision (HTML flow) |
| `POST` | `/api/plans/{id}/revise`| Owner / Admin | Submits feedback and creates new plan version (JSON API flow) |
| `GET` | `/tips` | Authenticated | Displays latest tip and tips history |
| `POST` | `/tips/generate` | Authenticated | Generates and saves a nutrition or recovery tip (HTML flow) |
| `GET` | `/api/tips` | Authenticated | Lists user's saved tips in JSON |
| `POST` | `/api/tips/generate` | Authenticated | Generates and saves a tip (JSON API flow) |
| `GET` | `/api/tips/{id}` | Owner / Admin | Returns specific tip details in JSON |
| `GET` | `/admin` | **Admin Only** | Protected dashboard: live DB counts, search, paginated users |
| `GET` | `/api/admin/overview` | **Admin Only** | Platform statistics JSON (users, plans, revisions, tips) |
| `GET` | `/api/admin/users` | **Admin Only** | Paginated users list JSON (omits password hashes/tokens) |
| `GET` | `/admin/users/{id}` | **Admin Only** | Inspects user profile snapshot, plan history, and tips |
| `GET` | `/api/admin/users/{id}`| **Admin Only** | Returns structured user inspection JSON |
| `GET` | `/admin/plans/{id}` | **Admin Only** | Inspects 7-day schedule, version tree, feedback, model badge |
| `GET` | `/api/admin/plans/{id}`| **Admin Only** | Returns structured plan inspection JSON |

---

## Beginner-Friendly Windows PowerShell Setup

Follow these exact commands to set up, initialize, and start FitBuddy:

### Step 1: Open PowerShell in Project Root
```powershell
cd "c:\Users\admin\.antigravity new"
```

### Step 2: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy restriction, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then re-run).*

### Step 3: Verify Installed Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Local Environment (`.env`)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### Step 5: Initialize SQLite Database
Initialize database tables non-destructively:
```powershell
python manage.py init-db
```

### Step 6: Create an Administrator Account
Create your local admin account:
```powershell
python manage.py create-admin --email admin@example.com --password AdminPassword123!
```

### Step 7: Start the FitBuddy Server
```powershell
python run.py
```

### Step 8: Access the Application in Your Browser
- **Application Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health Status**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Gemini Model Selection & API Key Configuration

FitBuddy uses the official Google GenAI Python SDK (`google-genai`).

### How to Configure Your Gemini API Key:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Open your local `.env` file in the project root:
   ```ini
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. Set your preferred model:
   ```ini
   GEMINI_WORKOUT_MODEL=gemini-2.5-flash
   GEMINI_TIP_MODEL=gemini-2.5-flash
   ```
4. Run the interactive diagnostic tool:
   ```powershell
   python manage.py gemini-diagnostics
   ```
   This will test your key, list all models accessible to your key, and execute a minimal 1-token test prompt without spending unnecessary quota.

### Development Mock Mode (No Key Required):
If you do not have an active Gemini API key, you can run FitBuddy in deterministic mock mode by setting:
```ini
GEMINI_MOCK_ENABLED=true
```
In mock mode:
- 7-day plans are generated using a deterministic, validated mock schedule.
- Tips are generated using structured mock tips across all 3 goals and 2 categories.
- Admin dashboard and plan headers clearly badge output as `⚙️ Mock Simulation Mode (mock-fitbuddy-v1)`.
- Live failures **never** silently fall back to mock data.

---

## Automated Testing & Smoke Diagnostics

FitBuddy includes a 70-test automated test suite.

### Run All Automated Tests:
```powershell
pytest -v
```

### Test Suite Structure:
- `tests/test_auth_and_profile.py`: Registration, login, logout, profile bounds, CSRF, account deletion.
- `tests/test_workout_plans.py`: 7-day schema validation, day order, duration/equipment validation, prompt injection defense, bounded repair, revision flow, version history, duplicate clicks.
- `tests/test_tips.py`: Tips across all 3 goals and 2 categories, debounce rate-limiting, persistence.
- `tests/test_admin.py`: RBAC, direct URL 403 access denial, live database telemetry, search, user/plan inspection, credential masking, read-only enforcement.
- `tests/test_ai_service.py`: PII minimization, key masking, missing credentials, transient retry backoff, safety blocks, malformed output detection.
- `tests/test_restart_persistence.py`: Validates SQLite data persistence and multi-user isolation across application restarts.
- `tests/test_cli_diagnostics.py`: CLI diagnostics command testing under unconfigured and configured states.
- `tests/test_health.py`: Health endpoint and static asset delivery.

---

## Three-Minute Demonstration Script

Follow this step-by-step walkthrough to demonstrate all three core scenarios:

### Part 1: Registration & Profile Setup (45 seconds)
1. Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. Click **Register** (`/register`) and create an account:
   - Email: `demo_athlete@fitbuddy.test`
   - Password: `AthletePassword123!`
3. You will be redirected to **My Profile** (`/profile`). Fill out your fitness profile:
   - Display Name: `Alex Runner`
   - Age: `28` (Scope: 18+)
   - Weight: `70.0` kg
   - Fitness Goal: `Muscle Gain`
   - Workout Intensity: `Medium`
   - Experience Level: `Intermediate`
   - Equipment: `Dumbbells`
   - Available Time: `45` minutes
   - Exercise Limitations: `Mild right shoulder impingement`
4. Click **Save Profile**. A green confirmation banner confirms persistence.

### Part 2: Scenario 1 — Generate a Personalized 7-Day Workout Plan (45 seconds)
1. Click **Workout Plans** (`/plans`) in the top navigation.
2. Click **Generate 7-Day Plan**.
3. Inspect the resulting 7-day schedule:
   - Note the plan title, goal, summary, and general guidance.
   - Verify all 7 days are uniquely numbered, chronological, and include warm-ups, dumbbell exercises (sets, reps, rest, instructions), cooldowns, and recovery notes.
   - Note the model badge: `Mock Simulation Mode` (or `Live AI Generation` if API key configured).

### Part 3: Scenario 2 — Revise Plan with User Feedback (45 seconds)
1. On the plan detail page (`/plans/{id}`), scroll to the **Request Plan Revision** card.
2. Enter feedback: `"Please replace shoulder pressing with bodyweight squats and add an extra rest day for my shoulder."`
3. Click **Submit Feedback & Revise Plan**.
4. The backend confirms ownership, sends feedback to Gemini, validates the revision, and saves it as **Version 2**.
5. Inspect the version timeline: click **Switch to v1** to verify that Version 1 remains intact. Switch back to **v2** to see the revision and the AI change explanation notice.

### Part 4: Scenario 3 — Daily Nutrition & Recovery Tips (30 seconds)
1. Click **Daily Tips** (`/tips`) in the top navigation.
2. Select **Nutrition** and click **Get Daily Tip**. A concise, actionable nutrition tip aligned with Muscle Gain appears.
3. Select **Recovery** and click **Get Daily Tip**. A recovery tip appears, and previous tips appear in the history list.

### Part 5: Restart Persistence & Admin Verification (15 seconds)
1. Stop the server (`Ctrl+C` in PowerShell) and restart it (`python run.py`).
2. Refresh the browser: all workout plans, revisions, profiles, and tips remain intact in SQLite!
3. Log in as admin (`admin@example.com`):
   - Access **Admin Dashboard** (`/admin`).
   - Observe live database counts matching the exact number of users, plans, revisions, and tips.
   - Inspect user `demo_athlete@fitbuddy.test` and view the version lineage tree and user feedback!

---

## Concise Architectural Explanation

FitBuddy’s architecture is built on six complementary pillars:

1. **FastAPI**:
   Provides an asynchronous, high-performance web foundation. Uses FastAPI dependencies (`get_db`, `get_current_user`, `require_admin_user`) for clean separation of concerns, automatic OpenAPI documentation, and role-based access control.
2. **Google Gemini GenAI SDK (`google-genai`)**:
   Communicates with Gemini models (`gemini-2.5-flash`) via the modern SDK. Requests are performed asynchronously with bounded retries and exponential backoff for transient errors, and never hold SQLite database locks while awaiting network responses.
3. **Structured Prompting & Structured Output Schemas**:
   Fitness profiles are sanitized through an `AnonymousFitnessProfile` schema (stripping emails and personal identifiers). The prompt instructs Gemini to return strict JSON adhering to `WorkoutPlanResponseSchema` or `FitnessTipResponseSchema`. If validation fails, a bounded repair loop provides the exact error feedback to Gemini to self-correct before rejecting the plan.
4. **Local SQLite with WAL Mode**:
   Local-first persistence using SQLite 3 with Write-Ahead Logging (`WAL` mode). This allows concurrent readers while writes occur, delivering durability across application restarts without complex external database infrastructure.
5. **Feedback & Version Lineage**:
   Revisions do not overwrite parent plans. When a user submits feedback, FitBuddy validates source plan ownership, creates a new `WorkoutPlan` row with incremented version and `parent_plan_id`, and creates a `PlanFeedback` row linking the two versions with timestamps.
6. **Jinja2 Server-Side Templating**:
   Server-rendered HTML templates utilizing standard auto-escaping to prevent Cross-Site Scripting (XSS). Combined with modern Vanilla CSS glassmorphism, responsive media queries, and semantic HTML5 elements.

---

## Honest Limitations & Future Improvements

### Current Limitations:
1. **Single-Process Local Deployment**:
   Designed for local evaluation and coursework review. Concurrency locks (`_generation_lock`) and rate-limiting dictionaries (`_last_tip_timestamps`) are in-memory per process. In a distributed multi-worker deployment, these would require Redis or a shared cache.
2. **Single SQLite Database File**:
   SQLite WAL mode is resilient for single-server setups, but concurrent write spikes during large multi-user evaluations could encounter transient table locks if not kept short.
3. **Read-Only Admin MVP**:
   The admin dashboard is strictly read-only for security and safety. It allows full telemetry, search, and inspection, but intentionally does not support user impersonation, password resetting, or bulk deletion from the web interface.
4. **Live Model Quota Dependency**:
   Live generation requires an active Google AI Studio API key and available quota. When unconfigured or quota-exhausted, FitBuddy reports transparent errors and does not pretend live generation succeeded.

### Planned Future Improvements:
- Multi-day workout tracking with checkbox completion logs.
- PDF/Print export for offline gym use.
- Multi-factor authentication (TOTP) for administrative accounts.
- Automated token bucket rate limiting backed by Redis.

---

## Final Verification & Status Table

| Milestone / Feature Area | Implementation Status | Test Coverage Status | Operational Notes |
|---|---|---|---|
| **1. Authentication & Security** | Completed | Verified (100%) | Argon2id, HTTP-only session cookies, HMAC CSRF, cascade account deletion |
| **2. Adult Fitness Profile** | Completed | Verified (100%) | 18+ age enforcement, weight/duration bounds, profile isolation |
| **3. 7-Day Plan Generator** | Completed | Verified (100%) | Pydantic v2 structured output, bounded repair loop, non-blocking DB |
| **4. Feedback & Plan Lineage** | Completed | Verified (100%) | Versioning (v1 &rarr; v2), feedback audit records, failure rollback |
| **5. Nutrition & Recovery Tips**| Completed | Verified (100%) | Both categories across all 3 goals, debounce rate limiting, history |
| **6. Protected Admin Dashboard**| Completed | Verified (100%) | Backend RBAC, direct URL 403 denial, live DB counts, search, inspection |
| **7. Restart Persistence** | Completed | Verified (100%) | SQLite WAL mode, multi-user isolation survives process restarts |
| **8. Diagnostics & CLI Tools** | Completed | Verified (100%) | `manage.py create-admin`, `manage.py gemini-diagnostics`, `init-db` |
| **9. Mock Mode Generation** | Completed | Verified (100%) | 70 automated tests pass with deterministic mock responses |
| **10. Live Gemini Generation** | Completed & Ready | Environment Pending | Fully implemented in code; unverified in this environment until user enters active `GEMINI_API_KEY` |

> [!NOTE]
> **Live Generation Verification Note**:
> All 70 unit and integration tests pass cleanly with mocked and deterministic simulated Gemini responses.
> Live API generation is fully implemented in `app/services/ai.py`, but has not been tested against Google servers in this session because `GEMINI_API_KEY` is not populated in `.env`.
> To run a live smoke test, enter your key in `.env` and execute `python manage.py gemini-diagnostics`.

---

## Troubleshooting Guide

### 1. "Authentication required" or Redirect Loop
- **Cause**: Browser cookie blocked or session expired.
- **Fix**: Clear cookies for `127.0.0.1:8000` or open in an Incognito/Private window. Ensure your browser allows cookies.

### 2. "GEMINI_API_KEY: Empty, missing, or default placeholder"
- **Cause**: `.env` file does not contain a valid key.
- **Fix**: Open `.env` and set `GEMINI_API_KEY=your_key`. If you want to test without an API key, set `GEMINI_MOCK_ENABLED=true`.

### 3. Port 8000 Already in Use
- **Cause**: Another local service is using port 8000.
- **Fix**: Change `PORT=8001` in `.env` and restart with `python run.py`.

### 4. PowerShell "Execution of scripts is disabled on this system"
- **Cause**: Default Windows PowerShell script execution policy.
- **Fix**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell.

### 5. "403 Forbidden: Admin authorization required"
- **Cause**: Accessing `/admin` with a regular user account or without logging in as admin.
- **Fix**: Log out, then create and log in as an administrator using `python manage.py create-admin`.
