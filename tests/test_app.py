import csv
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta

import src.app as app_module
from src.app import create_app
from src import data as data_module
from src.env_loader import load_env_file


STUDENT_EMAIL = "jhon.boiser@student.edu"
STUDENT_PASSWORD = "Student123!"
ADMIN_EMAIL = "benjie.samonte@semcds.edu"
ADMIN_PASSWORD = "Admin123!"


def seed_test_users() -> None:
    if not data_module.get_user_by_email(ADMIN_EMAIL):
        data_module.create_user(ADMIN_EMAIL, "Prof. Benjie Samonte", "admin", ADMIN_PASSWORD)
    if not data_module.get_user_by_email(STUDENT_EMAIL):
        data_module.create_user(STUDENT_EMAIL, "Jhon Boiser", "user", STUDENT_PASSWORD)


def build_app(tmp_path, monkeypatch):
    test_db_path = tmp_path / "semcds-test.db"
    monkeypatch.setattr(data_module, "DB_PATH", test_db_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    app_module._monitor_rooms.clear()
    app = create_app()
    app.config.update(TESTING=True)
    seed_test_users()
    return app


def extract_csrf_token(response) -> str:
    html = response.get_data(as_text=True)
    match = re.search(r'name="_csrf_token" value="([^"]+)"', html)
    assert match, "CSRF token was not rendered in the response."
    return match.group(1)


def login(client, *, role: str, email: str, password: str):
    response = client.get(f"/login?role={role}")
    csrf_token = extract_csrf_token(response)
    return client.post(
        "/login",
        data={
            "_csrf_token": csrf_token,
            "role": role,
            "email": email,
            "password": password,
        },
        follow_redirects=False,
    )


def extract_script_json(response, script_id: str) -> dict:
    html = response.get_data(as_text=True)
    match = re.search(rf'<script id="{re.escape(script_id)}" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match, f"Script JSON block {script_id!r} was not found."
    return json.loads(match.group(1))


def create_sample_quiz(
    *,
    title: str = "Network Security Basics",
    description: str = "Quiz used for route tests.",
    subject: str = "Security",
    quiz_code: str = "SEC123",
    monitoring_enabled: bool = True,
    status: str = "published",
    scheduled_start: str = "",
    scheduled_end: str = "",
    time_limit_minutes: int | None = None,
    creator_email: str = ADMIN_EMAIL,
    assigned_section: str = "",
) -> dict:
    admin = data_module.get_user_by_email(creator_email)
    assert admin is not None
    if time_limit_minutes is None:
        start = data_module.parse_schedule(scheduled_start)
        end = data_module.parse_schedule(scheduled_end)
        if start and end and end > start:
            time_limit_minutes = max(1, int((end - start).total_seconds() // 60))
        else:
            time_limit_minutes = 15

    quiz_id = data_module.create_or_update_quiz(
        quiz_id=None,
        creator_id=admin["id"],
        title=title,
        description=description,
        subject=subject,
        time_limit_minutes=time_limit_minutes,
        quiz_code=quiz_code,
        monitoring_enabled=monitoring_enabled,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_end,
        status=status,
        questions_payload=[
            {
                "question_text": "What does CSRF stand for?",
                "question_type": "multiple_choice",
                "options": [
                    "Cross-Site Request Forgery",
                    "Central Security Review Framework",
                    "Cross Session Response Filter",
                ],
                "correct_answer": "Cross-Site Request Forgery",
                "points": 1,
            }
        ],
        assigned_section=assigned_section,
    )
    quiz = data_module.get_quiz(quiz_id)
    assert quiz is not None
    return quiz


def test_database_backend_can_force_sqlite(monkeypatch):
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")

    assert data_module.database_backend() == "sqlite"
    assert data_module.using_supabase() is False


def test_sqlite_connection_uses_wal_for_live_updates(tmp_path, monkeypatch):
    test_db_path = tmp_path / "sqlite-live.db"
    monkeypatch.setattr(data_module, "DB_PATH", test_db_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")

    with data_module.get_connection() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert synchronous == 1
    assert busy_timeout == 5000
    assert foreign_keys == 1


def test_init_database_upgrades_legacy_quiz_table_before_creating_indexes(tmp_path, monkeypatch):
    test_db_path = tmp_path / "legacy-semcds.db"
    monkeypatch.setattr(data_module, "DB_PATH", test_db_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")

    with sqlite3.connect(test_db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                id VARCHAR(50) PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                full_name VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME NOT NULL
            );

            CREATE TABLE quizzes (
                id VARCHAR(50) PRIMARY KEY,
                creator_id VARCHAR(50) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                subject VARCHAR(255) NOT NULL,
                time_limit_minutes INT NOT NULL,
                status VARCHAR(20) NOT NULL,
                quiz_code VARCHAR(50) NOT NULL UNIQUE,
                monitoring_enabled BOOLEAN NOT NULL DEFAULT 0,
                scheduled_start DATETIME,
                scheduled_end DATETIME,
                created_at DATETIME NOT NULL
            );

            CREATE TABLE quiz_attempts (
                id VARCHAR(50) PRIMARY KEY,
                quiz_id VARCHAR(50) NOT NULL,
                student_id VARCHAR(50) NOT NULL,
                score INT NOT NULL DEFAULT 0,
                percentage DECIMAL(5,2) NOT NULL DEFAULT 0,
                status VARCHAR(20) NOT NULL,
                started_at DATETIME NOT NULL,
                submitted_at DATETIME,
                consent_given BOOLEAN NOT NULL DEFAULT 0
            );
            """
        )

    data_module.init_database()

    with data_module.get_connection() as connection:
        quiz_columns = {row[1] for row in connection.execute("PRAGMA table_info(quizzes)").fetchall()}
        attempt_columns = {row[1] for row in connection.execute("PRAGMA table_info(quiz_attempts)").fetchall()}
        index_names = {row[1] for row in connection.execute("PRAGMA index_list('quizzes')").fetchall()}

    assert "assigned_section" in quiz_columns
    assert "quiz_code" in attempt_columns
    assert "idx_quizzes_assigned_section" in index_names


def test_routes_load(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"SEMCDS" in response.data


def test_login_requires_csrf_token(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    response = client.post(
        "/login",
        data={
            "role": "user",
            "email": STUDENT_EMAIL,
            "password": STUDENT_PASSWORD,
        },
    )

    assert response.status_code == 400
    assert b"security token" in response.data


def test_login_page_renders_accessible_password_toggle(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    response = client.get("/login?role=user")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="toggle-password"' in html
    assert 'aria-controls="login-password"' in html
    assert 'aria-label="Show password"' in html
    assert "data-password-icon-show" in html
    assert "data-password-icon-hide" in html


def test_env_loader_uses_dotenv_without_overwriting_existing_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("SMTP_HOST=smtp.env.example\nSMTP_PORT=2525\n", encoding="utf-8")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.setenv("SMTP_PORT", "587")

    load_env_file(env_path)

    assert os.environ["SMTP_HOST"] == "smtp.env.example"
    assert os.environ["SMTP_PORT"] == "587"


def test_forgot_password_sends_reset_link_over_smtp_and_invalidates_used_token(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    sent_messages = []

    class DummySMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.used_starttls = False
            self.login_credentials = None

        def ehlo(self):
            return None

        def starttls(self, context=None):
            self.used_starttls = True
            return None

        def login(self, username, password):
            self.login_credentials = (username, password)
            return None

        def send_message(self, message, from_addr=None):
            sent_messages.append(
                {
                    "host": self.host,
                    "port": self.port,
                    "used_starttls": self.used_starttls,
                    "login": self.login_credentials,
                    "from_addr": from_addr,
                    "to": message["To"],
                    "subject": message["Subject"],
                    "body": message.get_content(),
                }
            )
            return None

        def quit(self):
            return None

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "smtp-user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-password")
    monkeypatch.setenv("SMTP_FROM", "no-reply@example.com")
    monkeypatch.setattr(app_module.smtplib, "SMTP", DummySMTP)
    monkeypatch.setattr(app_module.smtplib, "SMTP_SSL", DummySMTP)

    client = app.test_client()
    page = client.get("/forgot-password")
    csrf_token = extract_csrf_token(page)

    response = client.post(
        "/forgot-password",
        data={
            "_csrf_token": csrf_token,
            "email": STUDENT_EMAIL,
        },
        follow_redirects=False,
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "password reset link has been sent" in html
    assert len(sent_messages) == 1
    assert sent_messages[0]["host"] == "smtp.example.com"
    assert sent_messages[0]["port"] == 587
    assert sent_messages[0]["used_starttls"] is True
    assert sent_messages[0]["login"] == ("smtp-user@example.com", "smtp-password")
    assert sent_messages[0]["to"] == STUDENT_EMAIL
    assert sent_messages[0]["subject"] == "SEMCDS password reset"

    link_match = re.search(r"(/reset-password/[^\s]+)", sent_messages[0]["body"])
    assert link_match, "Password reset link was not present in the SMTP message body."
    reset_path = link_match.group(1)

    reset_page = client.get(reset_path)
    reset_csrf_token = extract_csrf_token(reset_page)
    reset_response = client.post(
        reset_path,
        data={
            "_csrf_token": reset_csrf_token,
            "password": "NewSecurePass123!",
            "confirm_password": "NewSecurePass123!",
        },
        follow_redirects=False,
    )

    updated_user = data_module.get_user_by_email(STUDENT_EMAIL)
    assert updated_user is not None
    assert reset_response.status_code == 302
    assert "/login" in reset_response.headers["Location"]
    assert updated_user["password_hash"] != "NewSecurePass123!"
    assert data_module.verify_password(updated_user, "NewSecurePass123!")
    assert not data_module.verify_password(updated_user, STUDENT_PASSWORD)

    used_token_response = client.get(reset_path)
    used_token_html = used_token_response.get_data(as_text=True)
    assert used_token_response.status_code == 200
    assert "already been used" in used_token_html

    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password="NewSecurePass123!",
    )
    assert login_response.status_code == 302


def test_forgot_password_sends_reset_link_over_resend_when_configured(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    resend_requests = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"id":"email_123"}'

    def fake_urlopen(request, timeout=0):
        resend_requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.setenv("EMAIL_DELIVERY_PROVIDER", "resend")
    monkeypatch.setenv("EMAIL_FROM", "verified@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_123")
    monkeypatch.setattr(app_module.urllib_request, "urlopen", fake_urlopen)

    client = app.test_client()
    page = client.get("/forgot-password")
    csrf_token = extract_csrf_token(page)

    response = client.post(
        "/forgot-password",
        data={
            "_csrf_token": csrf_token,
            "email": STUDENT_EMAIL,
        },
        follow_redirects=False,
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "password reset link has been sent" in html
    assert len(resend_requests) == 1
    assert resend_requests[0]["url"] == "https://api.resend.com/emails"
    assert resend_requests[0]["headers"]["Authorization"] == "Bearer re_test_123"
    assert resend_requests[0]["headers"]["User-agent"] == "SEMCDS/1.0"
    assert resend_requests[0]["body"]["from"] == "verified@example.com"
    assert resend_requests[0]["body"]["to"] == [STUDENT_EMAIL]
    assert resend_requests[0]["body"]["subject"] == "SEMCDS password reset"
    assert "/reset-password/" in resend_requests[0]["body"]["text"]


def test_forgot_password_sends_reset_link_over_brevo_when_configured(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    brevo_requests = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"messageId":"brevo_123"}'

    def fake_urlopen(request, timeout=0):
        brevo_requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("EMAIL_DELIVERY_PROVIDER", "brevo")
    monkeypatch.setenv("EMAIL_FROM", "verified@example.com")
    monkeypatch.setenv("EMAIL_FROM_NAME", "SEMCDS")
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test-123")
    monkeypatch.setattr(app_module.urllib_request, "urlopen", fake_urlopen)

    client = app.test_client()
    page = client.get("/forgot-password")
    csrf_token = extract_csrf_token(page)

    response = client.post(
        "/forgot-password",
        data={
            "_csrf_token": csrf_token,
            "email": STUDENT_EMAIL,
        },
        follow_redirects=False,
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "password reset link has been sent" in html
    assert len(brevo_requests) == 1
    assert brevo_requests[0]["url"] == "https://api.brevo.com/v3/smtp/email"
    assert brevo_requests[0]["headers"]["api-key"] == "xkeysib-test-123"
    assert brevo_requests[0]["headers"]["User-agent"] == "SEMCDS/1.0"
    assert brevo_requests[0]["body"]["sender"] == {"email": "verified@example.com", "name": "SEMCDS"}
    assert brevo_requests[0]["body"]["to"] == [{"email": STUDENT_EMAIL}]
    assert brevo_requests[0]["body"]["subject"] == "SEMCDS password reset"
    assert "/reset-password/" in brevo_requests[0]["body"]["textContent"]


def test_second_client_is_not_auto_logged_in(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    first_client = app.test_client()
    second_client = app.test_client()

    login_response = login(
        first_client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )

    assert login_response.status_code == 302
    second_response = second_client.get("/home", follow_redirects=False)

    assert second_response.status_code == 302
    assert second_response.headers["Location"].endswith("/login")


def test_student_dashboard_renders_session_avatar_url(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    with client.session_transaction() as session_state:
        session_state["avatar_url"] = "https://example.com/avatar.png"

    response = client.get("/StudentDashboard")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'src="https://example.com/avatar.png"' in html
    assert "avatar" in html


def test_user_management_renders_stored_user_avatar_url(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    assert data_module.set_user_avatar_url(student["id"], "https://example.com/student-avatar.png")

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/UserManagement")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'src="https://example.com/student-avatar.png"' in html


def test_submitted_student_can_still_view_results(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz()
    quiz_id = quiz["id"]
    question_id = quiz["questions"][0]["id"]
    client = app.test_client()

    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    take_quiz_response = client.get(f"/TakeQuiz?quizId={quiz_id}")
    take_quiz_html = take_quiz_response.get_data(as_text=True)
    assert take_quiz_response.status_code == 200

    csrf_token = extract_csrf_token(take_quiz_response)
    attempt_match = re.search(r'name="attempt_id" value="([^"]*)"', take_quiz_html)
    assert attempt_match, "Active attempt id was not rendered."
    attempt_id = attempt_match.group(1)

    submit_response = client.post(
        f"/TakeQuiz?quizId={quiz_id}",
        data={
            "_csrf_token": csrf_token,
            "quizId": quiz_id,
            "attempt_id": attempt_id,
            f"question_{question_id}": "Cross-Site Request Forgery",
            "consent_given": "on",
        },
        follow_redirects=False,
    )

    assert submit_response.status_code == 302
    result_response = client.get(submit_response.headers["Location"])

    assert result_response.status_code == 200
    assert b"Quiz Completed!" in result_response.data


def test_student_cannot_submit_quiz_with_unanswered_questions(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz()
    client = app.test_client()

    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    take_quiz_response = client.get(f"/TakeQuiz?quizId={quiz['id']}")
    take_quiz_html = take_quiz_response.get_data(as_text=True)
    assert take_quiz_response.status_code == 200

    csrf_token = extract_csrf_token(take_quiz_response)
    attempt_match = re.search(r'name="attempt_id" value="([^"]*)"', take_quiz_html)
    assert attempt_match, "Active attempt id was not rendered."

    submit_response = client.post(
        f"/TakeQuiz?quizId={quiz['id']}",
        data={
            "_csrf_token": csrf_token,
            "quizId": quiz["id"],
            "attempt_id": attempt_match.group(1),
            "consent_given": "on",
        },
        follow_redirects=False,
    )
    attempt = data_module.get_attempt(attempt_match.group(1))
    submit_html = submit_response.get_data(as_text=True)

    assert submit_response.status_code == 200
    assert "Answer all questions before submitting." in submit_html
    assert "Quiz Completed!" not in submit_html
    assert attempt is not None
    assert attempt["status"] == "in_progress"


def test_admin_cannot_open_student_take_quiz_route(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz_id = create_sample_quiz()["id"]
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/TakeQuiz?quizId={quiz_id}", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/home")


def test_passwordless_signin_route_was_removed(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    response = client.get("/signin/admin")

    assert response.status_code == 404


def test_quiz_manager_reports_filtered_counts_for_instructors(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    create_sample_quiz(title="Published Security Quiz", quiz_code="PUB123", status="published")
    create_sample_quiz(title="Draft Security Quiz", quiz_code="DRF123", status="draft")
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/QuizManager?status=published&q=Published")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "1 matching quiz out of 2" in html
    assert "View Results" in html


def test_quiz_manager_can_clear_only_current_instructor_created_quizzes(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    first_quiz = create_sample_quiz(title="Owned Quiz One", quiz_code="OWN101")
    second_quiz = create_sample_quiz(title="Owned Quiz Two", quiz_code="OWN102", status="draft")
    other_admin = data_module.create_user("other.admin@example.edu", "Other Admin", "admin", "Admin12345!")
    other_quiz = create_sample_quiz(
        title="Other Admin Quiz",
        quiz_code="OTH201",
        creator_email=other_admin["email"],
    )
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    page = client.get("/QuizManager")
    html = page.get_data(as_text=True)
    csrf_token = extract_csrf_token(page)

    assert page.status_code == 200
    assert "Apply Filters" in html
    assert "Clear My Quizzes" in html
    assert "Owned Quiz One" in html
    assert "Owned Quiz Two" in html
    assert "Other Admin Quiz" not in html

    clear_response = client.post(
        "/QuizAction",
        data={
            "_csrf_token": csrf_token,
            "action": "clear_all_created",
        },
        follow_redirects=False,
    )

    assert clear_response.status_code == 302
    assert data_module.get_quiz(first_quiz["id"]) is None
    assert data_module.get_quiz(second_quiz["id"]) is None
    assert data_module.get_quiz(other_quiz["id"]) is not None


def test_activity_monitor_uses_filtered_stats_and_review_status_badges(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="MON123", monitoring_enabled=True, status="published")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], False)
    data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=attempt_id,
        event_type="no_face_detected",
        event_description="Face left the camera frame.",
        flag_level="medium",
    )
    reviewed_log = data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=attempt_id,
        event_type="face_detected",
        event_description="Face returned to frame.",
        flag_level="low",
    )
    with data_module.get_connection() as connection:
        connection.execute("UPDATE activity_logs SET reviewed = 1 WHERE id = ?", (reviewed_log["id"],))

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(
        f"/ActivityMonitor?quizId={quiz['id']}&studentEmail={STUDENT_EMAIL.lower()}&reviewed=pending"
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<strong>1</strong>" in html
    assert "<span>Total Events</span>" in html
    assert "<span>Unreviewed</span>" in html
    assert "No Face Detected" in html
    assert 'status-badge pending">Pending</span>' in html


def test_quiz_results_renders_result_status_badges(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="RST123", status="published")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], True)
    answers = {quiz["questions"][0]["id"]: "Cross-Site Request Forgery"}
    data_module.finalize_quiz_attempt(attempt_id, answers, True)
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/QuizResults?quizId={quiz['id']}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'status-badge submitted">Submitted</span>' in html


def test_quiz_results_show_student_ranking_and_score_for_instructors(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="RNK123", status="published")
    top_student = data_module.get_user_by_email(STUDENT_EMAIL)
    second_student = data_module.create_user("ranking.second@example.edu", "Ranking Second", "user", "Rank12345!")
    active_student = data_module.create_user("ranking.active@example.edu", "Ranking Active", "user", "Rank12345!")
    assert top_student is not None

    top_attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], top_student["id"], True)
    second_attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], second_student["id"], True)
    data_module.ensure_quiz_attempt_in_progress(quiz["id"], active_student["id"], False)

    question_id = quiz["questions"][0]["id"]
    data_module.finalize_quiz_attempt(
        top_attempt_id,
        {question_id: "Cross-Site Request Forgery"},
        True,
    )
    data_module.finalize_quiz_attempt(
        second_attempt_id,
        {question_id: "Wrong Answer"},
        True,
    )

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/QuizResults?quizId={quiz['id']}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Submission Ranking" in html
    assert "Rank #1" in html
    assert "Jhon Boiser" in html
    assert "1/1" in html
    assert "100.0%" in html or "100%" in html
    assert "Ranking Second" in html
    assert "0/1" in html
    assert "50.0%" in html
    assert html.index("Jhon Boiser") < html.index("Ranking Second")
    assert "Ranking Active" in html
    assert "Still active" in html
    assert '<td>-</td>' in html


def test_quiz_results_can_export_csv(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="CSV123", status="published")
    top_student = data_module.get_user_by_email(STUDENT_EMAIL)
    second_student = data_module.create_user("csv.second@example.edu", "CSV Second", "user", "Csv12345!")
    active_student = data_module.create_user("csv.active@example.edu", "CSV Active", "user", "Csv12345!")
    assert top_student is not None

    top_attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], top_student["id"], True)
    second_attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], second_student["id"], True)
    data_module.ensure_quiz_attempt_in_progress(quiz["id"], active_student["id"], False)
    question_id = quiz["questions"][0]["id"]

    data_module.finalize_quiz_attempt(
        top_attempt_id,
        {question_id: "Cross-Site Request Forgery"},
        True,
    )
    data_module.finalize_quiz_attempt(
        second_attempt_id,
        {question_id: "Wrong Answer"},
        True,
    )
    data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=second_attempt_id,
        event_type="tab_switch",
        event_description="Student switched tabs during the quiz.",
        flag_level="medium",
    )

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/QuizResults/ExportCsv?quizId={quiz['id']}")
    rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment;" in response.headers.get("Content-Disposition", "")
    assert response.headers["Content-Disposition"].endswith('network-security-basics-results.csv"')
    assert len(rows) == 3
    assert rows[0]["Rank"] == "1"
    assert rows[0]["Name"] == "Jhon Boiser"
    assert rows[0]["Status"] == "Submitted"
    assert rows[0]["Score"] == "1"
    assert rows[0]["Total Points"] == "1"
    assert rows[1]["Name"] == "CSV Second"
    assert rows[1]["Flag Count"] == "1"
    assert rows[2]["Name"] == "CSV Active"
    assert rows[2]["Rank"] == ""
    assert rows[2]["Status"] == "In Progress"
    assert rows[2]["Submission Time"] == "Still active"
    assert rows[2]["Started At"] != ""
    assert rows[2]["Email"] == "csv.active@example.edu"


def test_student_dashboard_only_shows_open_and_unfinished_quizzes(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    now = datetime.now()
    open_start = (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
    open_end = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    future_start = (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    future_end = (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    closed_start = (now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    closed_end = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

    visible_quiz = create_sample_quiz(
        title="Open Quiz",
        quiz_code="OPEN1",
        scheduled_start=open_start,
        scheduled_end=open_end,
    )
    create_sample_quiz(
        title="Future Quiz",
        quiz_code="FUT123",
        scheduled_start=future_start,
        scheduled_end=future_end,
    )
    create_sample_quiz(
        title="Closed Quiz",
        quiz_code="CLS123",
        scheduled_start=closed_start,
        scheduled_end=closed_end,
    )
    completed_quiz = create_sample_quiz(
        title="Completed Quiz",
        quiz_code="DON123",
        scheduled_start=open_start,
        scheduled_end=open_end,
    )

    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(completed_quiz["id"], student["id"], True)
    answers = {completed_quiz["questions"][0]["id"]: "Cross-Site Request Forgery"}
    data_module.finalize_quiz_attempt(attempt_id, answers, True)

    client = app.test_client()
    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/StudentDashboard")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Open Quiz" in html
    assert "Future Quiz" not in html
    assert "Closed Quiz" not in html
    assert "Completed Quiz" not in html
    assert "Active Quizzes" in html
    assert "Showing only quizzes that are active right now and still available for you to take" in html
    assert f'href="/TakeQuiz?quizId={visible_quiz["id"]}"' in html
    assert "Opens:" in html
    assert "Closes:" in html
    assert "Quiz History" not in html
    assert "Passed (" not in html


def test_completed_quiz_stays_locked_after_quiz_code_changes(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(title="One Attempt Quiz", quiz_code="LOCK1")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    admin = data_module.get_user_by_email(ADMIN_EMAIL)
    assert student is not None
    assert admin is not None

    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], True)
    answers = {quiz["questions"][0]["id"]: "Cross-Site Request Forgery"}
    data_module.finalize_quiz_attempt(attempt_id, answers, True)

    data_module.create_or_update_quiz(
        quiz_id=quiz["id"],
        creator_id=admin["id"],
        title=quiz["title"],
        description=quiz["description"],
        subject=quiz["subject"],
        time_limit_minutes=quiz["time_limit_minutes"],
        quiz_code="LOCK2",
        monitoring_enabled=quiz["monitoring_enabled"],
        scheduled_start=quiz["scheduled_start"] or "",
        scheduled_end=quiz["scheduled_end"] or "",
        status=quiz["status"],
        questions_payload=quiz["questions"],
    )
    updated_quiz = data_module.get_quiz(quiz["id"])
    assert updated_quiz is not None

    access_allowed, access_message = data_module.quiz_access_state(updated_quiz, student["id"])

    assert access_allowed is False
    assert access_message == "You have already completed this quiz."


def test_live_view_hides_latest_detection_status_panel(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(title="Live Monitor Quiz", quiz_code="LIVE1", monitoring_enabled=True, status="published")
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/ActivityMonitor?live=1&liveQuizId={quiz['id']}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Live Camera Feed" in html
    assert "Latest Detection Status" not in html


def test_live_view_stays_on_live_page_without_active_quiz(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/ActivityMonitor?live=1")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Live View</h1>" in html
    assert "No Live Quiz Available" in html
    assert "Review flagged activity events during quizzes. Flags are for review only." not in html


def test_expired_published_quiz_is_auto_closed_on_next_request(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    now = datetime.now()
    quiz = create_sample_quiz(
        title="Expired Quiz",
        quiz_code="EXP123",
        status="published",
        scheduled_start=(now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
        scheduled_end=(now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"),
    )
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/QuizManager?status=closed")
    html = response.get_data(as_text=True)
    refreshed_quiz = data_module.get_quiz(quiz["id"])

    assert response.status_code == 200
    assert refreshed_quiz is not None
    assert refreshed_quiz["status"] == "closed"
    assert "Expired Quiz" in html
    assert all(item["id"] != quiz["id"] for item in data_module.open_quizzes())


def test_take_quiz_countdown_uses_remaining_attempt_time(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    now = datetime.now()
    quiz = create_sample_quiz(
        title="Timed Window Quiz",
        quiz_code="TIME1",
        status="published",
        scheduled_start=(now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),
        scheduled_end=(now + timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M"),
    )
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], True)
    with data_module.get_connection() as connection:
        connection.execute(
            "UPDATE quiz_attempts SET started_at = ? WHERE id = ?",
            (((now - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M")), attempt_id),
        )

    client = app.test_client()
    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/TakeQuiz?quizId={quiz['id']}")
    quiz_meta = extract_script_json(response, "quiz-meta")

    assert response.status_code == 200
    assert 0 < int(quiz_meta["countdownSecondsStart"]) <= 12 * 60
    assert int(quiz_meta["countdownSecondsStart"]) < int(quiz["time_limit_minutes"]) * 60


def test_create_quiz_defaults_to_zero_time_limit_until_schedule_is_set(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    page = client.get("/CreateQuiz")
    html = page.get_data(as_text=True)
    csrf_token = extract_csrf_token(page)

    assert page.status_code == 200
    assert 'name="assigned_section"' in html
    assert 'id="time-limit-minutes" name="time_limit_minutes" value="0"' in html
    assert 'id="time-limit-display" value="0 minutes"' in html

    response = client.post(
        "/CreateQuiz",
        data={
            "_csrf_token": csrf_token,
            "title": "Zero Default Quiz",
            "description": "No schedule yet.",
            "subject": "General",
            "quiz_code": "ZERO01",
            "time_limit_minutes": "0",
            "scheduled_start": "",
            "scheduled_end": "",
            "questions_payload": json.dumps(
                [
                    {
                        "question_text": "Sample question?",
                        "question_type": "multiple_choice",
                        "options": ["A", "B"],
                        "correct_answer": "A",
                        "points": 1,
                    }
                ]
            ),
            "action": "draft",
        },
        follow_redirects=False,
    )

    created_quiz = data_module.get_quiz_by_code("ZERO01")

    assert response.status_code == 302
    assert created_quiz is not None
    assert created_quiz["time_limit_minutes"] == 0


def test_student_dashboard_hides_quizzes_for_other_sections(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    now = datetime.now()
    open_start = (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
    open_end = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    data_module.update_user(
        student["id"],
        student["email"],
        student["full_name"],
        student["role"],
        None,
        section_name="BSIT 1A",
    )

    visible_quiz = create_sample_quiz(
        title="BSIT Quiz",
        quiz_code="BSIT1A",
        scheduled_start=open_start,
        scheduled_end=open_end,
        assigned_section="BSIT 1A",
    )
    create_sample_quiz(
        title="BSCS Quiz",
        quiz_code="BSCS1B",
        scheduled_start=open_start,
        scheduled_end=open_end,
        assigned_section="BSCS 1B",
    )

    client = app.test_client()
    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/StudentDashboard")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "BSIT Quiz" in html
    assert "BSCS Quiz" not in html
    assert f'href="/TakeQuiz?quizId={visible_quiz["id"]}"' in html
    assert "Section: BSIT 1A" in html


def test_join_quiz_blocks_students_from_other_sections(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    data_module.update_user(
        student["id"],
        student["email"],
        student["full_name"],
        student["role"],
        None,
        section_name="BSIT 1A",
    )
    create_sample_quiz(
        title="Restricted Quiz",
        quiz_code="REST01",
        assigned_section="BSCS 1B",
        status="published",
    )

    client = app.test_client()
    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/JoinQuiz?code=REST01")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Restricted Quiz" in html
    assert "Section / Class: BSCS 1B" in html
    assert "This quiz is only available to BSCS 1B. Your account is enrolled in BSIT 1A." in html
    assert "Quiz Locked" in html


def test_zero_default_time_limit_does_not_auto_submit_unscheduled_quiz(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(
        title="Unscheduled Quiz",
        quiz_code="OPEN00",
        scheduled_start="",
        scheduled_end="",
        time_limit_minutes=0,
    )
    client = app.test_client()

    login_response = login(
        client,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/TakeQuiz?quizId={quiz['id']}", follow_redirects=False)
    html = response.get_data(as_text=True)
    quiz_meta = extract_script_json(response, "quiz-meta")

    assert response.status_code == 200
    assert "Quiz Completed!" not in html
    assert "No limit" in html
    assert quiz_meta["countdownSecondsStart"] is None


def test_activity_monitor_shows_only_current_students_for_open_quiz(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    now = datetime.now()
    quiz = create_sample_quiz(
        title="Current Activity Quiz",
        quiz_code="CURR1",
        monitoring_enabled=True,
        status="published",
        scheduled_start=(now - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M"),
        scheduled_end=(now + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M"),
    )
    current_student = data_module.get_user_by_email(STUDENT_EMAIL)
    historical_student = data_module.create_user("history.student@example.edu", "History Student", "user", "History123!")
    assert current_student is not None

    current_attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], current_student["id"], False)
    historical_attempt = data_module.ensure_quiz_attempt_in_progress(quiz["id"], historical_student["id"], False)
    answers = {quiz["questions"][0]["id"]: "Cross-Site Request Forgery"}
    data_module.finalize_quiz_attempt(historical_attempt, answers, False)
    data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=historical_attempt,
        event_type="no_face_detected",
        event_description="Historical log for a finished attempt.",
        flag_level="medium",
    )
    app_module._monitor_rooms[f"monitor:{quiz['id']}"] = {
        "student-sid": {
            "sid": "student-sid",
            "quiz_id": quiz["id"],
            "role": "user",
            "user_id": current_student["id"],
            "attempt_id": current_attempt_id,
            "display_name": current_student["full_name"],
            "email": current_student["email"],
            "camera_on": False,
        }
    }

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(f"/ActivityMonitor?quizId={quiz['id']}")
    html = response.get_data(as_text=True)
    api_response = client.get(f"/api/quiz/{quiz['id']}/in-progress-students")
    api_payload = api_response.get_json()

    assert response.status_code == 200
    assert f'data-student-key="{STUDENT_EMAIL.lower()}"' in html
    assert 'data-student-key="history.student@example.edu"' not in html
    assert "Joined quiz" in html
    assert api_response.status_code == 200
    assert [student["student_name"] for student in api_payload["students"]] == [current_student["full_name"]]
    assert api_payload["students"][0]["latest_event"] == "Joined quiz"


def test_live_monitor_broadcasts_student_camera_status_to_instructor(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    if not app_module.socketio:
        return

    now = datetime.now()
    quiz = create_sample_quiz(
        title="Realtime Camera Quiz",
        quiz_code="RTC001",
        monitoring_enabled=True,
        status="published",
        scheduled_start=(now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M"),
        scheduled_end=(now + timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M"),
    )
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], True)

    admin_http = app.test_client()
    student_http = app.test_client()
    admin_login = login(
        admin_http,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    student_login = login(
        student_http,
        role="user",
        email=STUDENT_EMAIL,
        password=STUDENT_PASSWORD,
    )
    assert admin_login.status_code == 302
    assert student_login.status_code == 302

    admin_socket = app_module.socketio.test_client(app, flask_test_client=admin_http)
    student_socket = app_module.socketio.test_client(app, flask_test_client=student_http)

    assert admin_socket.is_connected()
    assert student_socket.is_connected()

    admin_socket.emit("join_monitor_room", {"quizId": quiz["id"], "role": "admin", "cameraOn": False})
    admin_events = admin_socket.get_received()
    assert any(event["name"] == "room_snapshot" for event in admin_events)
    assert not any(event["name"] == "monitor_error" for event in admin_events)

    student_socket.emit(
        "join_monitor_room",
        {"quizId": quiz["id"], "role": "user", "attemptId": attempt_id, "cameraOn": False},
    )
    student_events = student_socket.get_received()
    admin_events = admin_socket.get_received()

    assert any(event["name"] == "room_snapshot" for event in student_events)
    joined_event = next((event for event in admin_events if event["name"] == "participant_joined"), None)
    assert joined_event is not None
    assert joined_event["args"][0]["attempt_id"] == attempt_id
    assert joined_event["args"][0]["camera_on"] is False

    student_socket.emit("set_camera_status", {"quizId": quiz["id"], "cameraOn": True})
    admin_events = admin_socket.get_received()
    updated_event = next((event for event in admin_events if event["name"] == "participant_updated"), None)

    assert updated_event is not None
    assert updated_event["args"][0]["attempt_id"] == attempt_id
    assert updated_event["args"][0]["camera_on"] is True

    student_socket.disconnect()
    admin_socket.disconnect()


def test_activity_monitor_renders_alt_tab_attempt_labels(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="ALT123", monitoring_enabled=True, status="published")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], False)
    data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=attempt_id,
        event_type="alt_tab_attempt",
        event_description="Student pressed Alt+Tab while taking the quiz.",
        flag_level="high",
    )

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get(
        f"/ActivityMonitor?quizId={quiz['id']}&studentEmail={STUDENT_EMAIL.lower()}"
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Alt+Tab Attempt" in html


def test_user_management_supports_create_update_and_delete(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    page = client.get("/UserManagement")
    csrf_token = extract_csrf_token(page)

    create_response = client.post(
        "/UserManagement/Save",
        data={
            "_csrf_token": csrf_token,
            "full_name": "Managed Student",
            "email": "managed.student@example.edu",
            "role": "user",
            "section_name": "BSIT 2A",
            "password": "Managed123!",
        },
        follow_redirects=False,
    )
    created_user = data_module.get_user_by_email("managed.student@example.edu")

    assert create_response.status_code == 302
    assert created_user is not None
    assert created_user["full_name"] == "Managed Student"
    assert created_user["section_name"] == "BSIT 2A"

    update_response = client.post(
        "/UserManagement/Save",
        data={
            "_csrf_token": csrf_token,
            "user_id": created_user["id"],
            "full_name": "Managed Instructor",
            "email": "managed.instructor@example.edu",
            "role": "admin",
            "section_name": "BSIT 2A",
            "password": "Updated123!",
        },
        follow_redirects=False,
    )
    updated_user = data_module.get_user_by_email("managed.instructor@example.edu")

    assert update_response.status_code == 302
    assert updated_user is not None
    assert updated_user["role"] == "admin"
    assert updated_user["full_name"] == "Managed Instructor"
    assert updated_user["section_name"] == ""
    assert data_module.verify_password(updated_user, "Managed123!")
    assert not data_module.verify_password(updated_user, "Updated123!")

    delete_response = client.post(
        "/UserManagement/Delete",
        data={
            "_csrf_token": csrf_token,
            "user_id": updated_user["id"],
        },
        follow_redirects=False,
    )

    assert delete_response.status_code == 302
    assert data_module.get_user_by_id(updated_user["id"]) is None


def test_dashboard_shows_quiz_on_selected_day_within_schedule_span(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(
        title="Overnight Networking Exam",
        quiz_code="OVR111",
        status="published",
        monitoring_enabled=False,
        scheduled_start="2026-05-10 23:30",
        scheduled_end="2026-05-11 01:00",
    )
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/Dashboard?month=2026-05&day=2026-05-11")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert quiz["title"] in html
    assert "May 11, 2026" in html


def test_dashboard_schedule_api_returns_selected_day_quizzes(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(
        title="Routing Practical Exam",
        quiz_code="API222",
        status="published",
        monitoring_enabled=False,
        scheduled_start="2026-05-15 08:00",
        scheduled_end="2026-05-15 10:00",
    )
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/api/dashboard/schedule?day=2026-05-15")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["day"] == "2026-05-15"
    assert payload["month"] == "2026-05"
    assert payload["quizzes"][0]["id"] == quiz["id"]
    assert payload["quizzes"][0]["title"] == quiz["title"]


def test_dashboard_day_selection_overrides_stale_month_param(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(
        title="Switching Final Exam",
        quiz_code="MON333",
        status="published",
        monitoring_enabled=False,
        scheduled_start="2026-05-20 13:00",
        scheduled_end="2026-05-20 14:00",
    )
    client = app.test_client()

    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    response = client.get("/Dashboard?month=2026-04&day=2026-05-20")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert quiz["title"] in html
    assert "May 2026" in html


def test_dashboard_reset_clears_attempts_responses_and_activity_logs(tmp_path, monkeypatch):
    app = build_app(tmp_path, monkeypatch)
    quiz = create_sample_quiz(quiz_code="RSTALL", status="published")
    student = data_module.get_user_by_email(STUDENT_EMAIL)
    assert student is not None
    attempt_id = data_module.ensure_quiz_attempt_in_progress(quiz["id"], student["id"], True)
    answers = {quiz["questions"][0]["id"]: "Cross-Site Request Forgery"}
    data_module.finalize_quiz_attempt(attempt_id, answers, True)
    data_module.create_activity_log(
        quiz_id=quiz["id"],
        attempt_id=attempt_id,
        event_type="face_detected",
        event_description="Flag before reset.",
        flag_level="low",
    )

    client = app.test_client()
    login_response = login(
        client,
        role="admin",
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
    )
    assert login_response.status_code == 302

    page = client.get("/Dashboard")
    csrf_token = extract_csrf_token(page)
    reset_response = client.post(
        "/Dashboard/ResetData",
        data={"_csrf_token": csrf_token},
        follow_redirects=False,
    )

    with data_module.get_connection() as connection:
        attempts_total = connection.execute("SELECT COUNT(*) AS total FROM quiz_attempts").fetchone()["total"]
        responses_total = connection.execute("SELECT COUNT(*) AS total FROM student_responses").fetchone()["total"]
        flags_total = connection.execute("SELECT COUNT(*) AS total FROM activity_logs").fetchone()["total"]

    assert reset_response.status_code == 302
    assert attempts_total == 0
    assert responses_total == 0
    assert flags_total == 0
