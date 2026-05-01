from __future__ import annotations

import calendar
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .env_loader import load_env_file

try:
    from werkzeug.security import check_password_hash, generate_password_hash
except Exception:  # pragma: no cover - Werkzeug is expected with Flask, but keep a clear fallback.
    def generate_password_hash(password: str) -> str:
        raise RuntimeError("Werkzeug security helpers are not available.")


    def check_password_hash(_hash: str, _password: str) -> bool:
        raise RuntimeError("Werkzeug security helpers are not available.")


load_env_file()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "database" / "semcds.db")))
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

FEATURES = [
    "AI-assisted quiz generation from PDF and text references",
    "Real-time suspicious activity tracking during exam sessions",
    "Instructor dashboards with result analytics and flag review",
    "Student-friendly quiz flow with timer, consent, and instant feedback",
]

USER_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VALID_USER_ROLES = {"admin", "user"}

DEFAULT_USERS = [
    {
        "id": "user-admin-001",
        "email": "benjie.samonte@semcds.edu",
        "full_name": "Prof. Benjie Samonte",
        "role": "admin",
        "password": "Admin123!",
        "created_at": "2026-03-12 08:00",
    },
    {
        "id": "user-admin-002",
        "email": "23-78520@g.batstate-u.edu.ph",
        "full_name": "Lira Angela Ata",
        "role": "admin",
        "password": "Ata23-78520",
        "created_at": "2026-03-12 08:00",
    },
    {
        "id": "user-student-001",
        "email": "jhon.boiser@student.edu",
        "full_name": "Jhon Boiser",
        "role": "user",
        "password": "Student123!",
        "created_at": "2026-03-17 09:15",
    },
]

LEGACY_REMOVED_USER_IDS = [
    "user-student-002",
]

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_API_URL = f"{SUPABASE_URL.rstrip('/')}/rest/v1" if SUPABASE_URL else ""


def using_supabase() -> bool:
    return bool(SUPABASE_API_URL and SUPABASE_SERVICE_ROLE_KEY)


def _sb_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _sb_request(method: str, table_name: str, *, filters: dict | None = None, payload: dict | list[dict] | None = None) -> list[dict]:
    if not using_supabase():
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    url = f"{SUPABASE_API_URL}/{table_name}"
    if filters:
        query_pairs = []
        for key, value in filters.items():
            if isinstance(value, bool):
                encoded_value = f"eq.{str(value).lower()}"
            else:
                encoded_value = f"eq.{value}"
            query_pairs.append((key, encoded_value))
        query_string = "&".join(
            f"{urllib_parse.quote(str(key))}={urllib_parse.quote(str(value), safe='.,()')}"
            for key, value in query_pairs
        )
        url = f"{url}?{query_string}"

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib_request.Request(url, data=data, headers=_sb_headers(), method=method)
    try:
        with urllib_request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise RuntimeError(f"Supabase request failed: {error_body}") from exc

    if not body.strip():
        return []
    parsed = json.loads(body)
    return parsed if isinstance(parsed, list) else [parsed]


def _sb_table(table_name: str):
    return table_name


def _sb_select(table_name: str, filters: dict | None = None) -> list[dict]:
    return [dict(row) for row in _sb_request("GET", table_name, filters=filters)]


def _sb_insert(table_name: str, payload: dict | list[dict]) -> None:
    _sb_request("POST", table_name, payload=payload)


def _sb_update(table_name: str, payload: dict, filters: dict) -> None:
    _sb_request("PATCH", table_name, filters=filters, payload=payload)


def _sb_delete(table_name: str, filters: dict) -> None:
    _sb_request("DELETE", table_name, filters=filters)


def _sb_delete_many(table_name: str, field_name: str, values: list[str]) -> None:
    if not values:
        return
    for value in values:
        _sb_delete(table_name, {field_name: value})


def _sort_rows(rows: list[dict], key_name: str, reverse: bool = False) -> list[dict]:
    return sorted(rows, key=lambda item: str(item.get(key_name, "")), reverse=reverse)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _normalize_section_name(section_name: str | None) -> str:
    return " ".join(str(section_name or "").split()).casefold()


def _text_to_datetime(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    value = str(raw_value).strip()
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    if using_supabase():
        existing_users = _sb_select("users")
        existing_by_id = {str(row.get("id", "")): row for row in existing_users}
        for user in DEFAULT_USERS:
            payload = {
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"],
                "password_hash": generate_password_hash(user["password"]),
                "created_at": user["created_at"],
            }
            if user["id"] in existing_by_id:
                _sb_update("users", payload, {"id": user["id"]})
            else:
                _sb_insert("users", {"id": user["id"], **payload})

        for removed_user_id in LEGACY_REMOVED_USER_IDS:
            if removed_user_id in existing_by_id:
                _sb_delete("users", {"id": removed_user_id})
        return

    with get_connection() as connection:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        connection.executescript(schema)
        user_columns = [row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()]
        if "section_name" not in user_columns:
            connection.execute("ALTER TABLE users ADD COLUMN section_name VARCHAR(255) NOT NULL DEFAULT ''")
        if "avatar_url" not in user_columns:
            connection.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT NOT NULL DEFAULT ''")
        quiz_columns = [row[1] for row in connection.execute("PRAGMA table_info(quizzes)").fetchall()]
        if "assigned_section" not in quiz_columns:
            connection.execute("ALTER TABLE quizzes ADD COLUMN assigned_section VARCHAR(255) NOT NULL DEFAULT ''")
        existing_columns = [row[1] for row in connection.execute("PRAGMA table_info(quiz_attempts)").fetchall()]
        if "quiz_code" not in existing_columns:
            connection.execute("ALTER TABLE quiz_attempts ADD COLUMN quiz_code VARCHAR(50) NOT NULL DEFAULT ''")
        count = connection.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"]
        if count == 0:
            connection.executemany(
                """
                INSERT INTO users (id, email, full_name, role, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        user["id"],
                        user["email"],
                        user["full_name"],
                        user["role"],
                        generate_password_hash(user["password"]),
                        user["created_at"],
                    )
                    for user in DEFAULT_USERS
                ],
            )
        else:
            for user in DEFAULT_USERS:
                existing_user = connection.execute(
                    "SELECT id FROM users WHERE id = ?",
                    (user["id"],),
                ).fetchone()
                if existing_user:
                    connection.execute(
                        """
                        UPDATE users
                        SET email = ?, full_name = ?, role = ?, password_hash = ?, created_at = ?
                        WHERE id = ?
                        """,
                        (
                            user["email"],
                            user["full_name"],
                            user["role"],
                            generate_password_hash(user["password"]),
                            user["created_at"],
                            user["id"],
                        ),
                    )
                else:
                    existing_email_user = connection.execute(
                        "SELECT id FROM users WHERE lower(email) = ?",
                        (user["email"].strip().lower(),),
                    ).fetchone()
                    if existing_email_user:
                        connection.execute(
                            """
                            UPDATE users
                            SET full_name = ?, role = ?, password_hash = ?, created_at = ?
                            WHERE id = ?
                            """,
                            (
                                user["full_name"],
                                user["role"],
                                generate_password_hash(user["password"]),
                                user["created_at"],
                                existing_email_user["id"],
                            ),
                        )
                    else:
                        connection.execute(
                            """
                            INSERT INTO users (id, email, full_name, role, password_hash, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                user["id"],
                                user["email"],
                                user["full_name"],
                                user["role"],
                                generate_password_hash(user["password"]),
                                user["created_at"],
                            ),
                        )

            for removed_user_id in LEGACY_REMOVED_USER_IDS:
                connection.execute("DELETE FROM users WHERE id = ?", (removed_user_id,))


def next_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


def parse_schedule(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw_value, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def format_schedule(raw_value: str | None) -> str:
    parsed = parse_schedule(raw_value)
    if not parsed:
        return "Not scheduled"
    return parsed.strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")


def verify_password(user: dict | None, password: str) -> bool:
    if not user:
        return False
    return check_password_hash(user["password_hash"], password)


def get_user(role: str) -> dict | None:
    if using_supabase():
        rows = [row for row in _sb_select("users") if row.get("role") == role]
        rows = _sort_rows(rows, "created_at")
        return rows[0] if rows else None

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users WHERE role = ? ORDER BY created_at LIMIT 1",
            (role,),
        ).fetchone()
    return dict(row) if row else None


def get_users() -> list[dict]:
    if using_supabase():
        return _sort_rows(_sb_select("users"), "created_at")

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users ORDER BY datetime(created_at), rowid"
        ).fetchall()
    return [dict(row) for row in rows]


def user_record_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}

    def ensure_bucket(user_id: str) -> dict[str, int]:
        return counts.setdefault(user_id, {"owned_quizzes": 0, "quiz_attempts": 0})

    if using_supabase():
        for quiz in _sb_select("quizzes"):
            creator_id = str(quiz.get("creator_id", "")).strip()
            if creator_id:
                ensure_bucket(creator_id)["owned_quizzes"] += 1
        for attempt in _sb_select("quiz_attempts"):
            student_id = str(attempt.get("student_id", "")).strip()
            if student_id:
                ensure_bucket(student_id)["quiz_attempts"] += 1
        return counts

    with get_connection() as connection:
        owned_rows = connection.execute(
            """
            SELECT creator_id AS user_id, COUNT(*) AS total
            FROM quizzes
            GROUP BY creator_id
            """
        ).fetchall()
        attempt_rows = connection.execute(
            """
            SELECT student_id AS user_id, COUNT(*) AS total
            FROM quiz_attempts
            GROUP BY student_id
            """
        ).fetchall()

    for row in owned_rows:
        ensure_bucket(str(row["user_id"]))["owned_quizzes"] = int(row["total"])
    for row in attempt_rows:
        ensure_bucket(str(row["user_id"]))["quiz_attempts"] = int(row["total"])
    return counts


def get_user_by_id(user_id: str) -> dict | None:
    if using_supabase():
        rows = _sb_select("users", {"id": user_id})
        return rows[0] if rows else None

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    if using_supabase():
        normalized_email = _normalize_email(email)
        rows = [row for row in _sb_select("users") if row.get("email", "").strip().lower() == normalized_email]
        return rows[0] if rows else None

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users WHERE lower(email) = ?",
            (_normalize_email(email),),
        ).fetchone()
    return dict(row) if row else None


def create_user(email: str, full_name: str, role: str, password: str, section_name: str = "") -> dict:
    normalized_email = _normalize_email(email)
    role = str(role).strip().lower()
    full_name = full_name.strip() or normalized_email
    clean_section_name = " ".join(str(section_name or "").split()) if role == "user" else ""
    if not normalized_email or not USER_EMAIL_PATTERN.match(normalized_email):
        raise ValueError("Enter a valid email address.")
    if role not in VALID_USER_ROLES:
        raise ValueError("Choose a valid role.")
    if not password:
        raise ValueError("A password is required.")
    if get_user_by_email(normalized_email):
        raise ValueError("A user with this email already exists.")
    created_at = _now_stamp()
    if using_supabase():
        user_id = next_id("user")
        payload = {
            "id": user_id,
            "email": normalized_email,
            "full_name": full_name,
            "role": role,
            "section_name": clean_section_name,
            "password_hash": generate_password_hash(password),
            "created_at": created_at,
        }
        _sb_insert("users", payload)
        return payload

    with get_connection() as connection:
        user_id = next_id("user")
        connection.execute(
            "INSERT INTO users (id, email, full_name, role, section_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                normalized_email,
                full_name,
                role,
                clean_section_name,
                generate_password_hash(password),
                created_at,
            ),
        )
        row = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else {}


def update_user(
    user_id: str,
    email: str,
    full_name: str,
    role: str,
    password: str | None = None,
    section_name: str = "",
) -> dict:
    target_user = get_user_by_id(user_id)
    if not target_user:
        raise ValueError("User not found.")

    normalized_email = _normalize_email(email)
    normalized_role = str(role).strip().lower()
    clean_name = full_name.strip()
    clean_section_name = " ".join(str(section_name or "").split()) if normalized_role == "user" else ""
    if not normalized_email or not USER_EMAIL_PATTERN.match(normalized_email):
        raise ValueError("Enter a valid email address.")
    if not clean_name:
        raise ValueError("Full name is required.")
    if normalized_role not in VALID_USER_ROLES:
        raise ValueError("Choose a valid role.")

    existing_email_user = get_user_by_email(normalized_email)
    if existing_email_user and existing_email_user["id"] != user_id:
        raise ValueError("A user with this email already exists.")

    payload = {
        "email": normalized_email,
        "full_name": clean_name,
        "role": normalized_role,
        "section_name": clean_section_name,
    }
    if password:
        payload["password_hash"] = generate_password_hash(password)

    if using_supabase():
        _sb_update("users", payload, {"id": user_id})
        updated = get_user_by_id(user_id)
        return updated or {}

    with get_connection() as connection:
        assignments = ["email = ?", "full_name = ?", "role = ?", "section_name = ?"]
        values: list[str] = [normalized_email, clean_name, normalized_role, clean_section_name]
        if password:
            assignments.append("password_hash = ?")
            values.append(generate_password_hash(password))
        values.append(user_id)
        connection.execute(
            f"UPDATE users SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        row = connection.execute(
            "SELECT id, email, full_name, role, section_name, avatar_url, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else {}


def set_user_password(user_id: str, password: str) -> bool:
    if not user_id:
        return False

    password_hash = generate_password_hash(password)
    if using_supabase():
        existing = _sb_select("users", {"id": user_id})
        if not existing:
            return False
        _sb_update("users", {"password_hash": password_hash}, {"id": user_id})
        return True

    with get_connection() as connection:
        result = connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )
        return result.rowcount > 0


def set_user_avatar_url(user_id: str, avatar_url: str) -> bool:
    if not user_id:
        return False

    clean_avatar_url = str(avatar_url or "").strip()
    if using_supabase():
        try:
            existing = _sb_select("users", {"id": user_id})
            if not existing:
                return False
            _sb_update("users", {"avatar_url": clean_avatar_url}, {"id": user_id})
            return True
        except RuntimeError:
            return False

    with get_connection() as connection:
        result = connection.execute(
            "UPDATE users SET avatar_url = ? WHERE id = ?",
            (clean_avatar_url, user_id),
        )
        return result.rowcount > 0


def delete_user_by_email(email: str) -> None:
    normalized_email = _normalize_email(email)
    if using_supabase():
        _sb_delete("users", {"email": normalized_email})
        return

    with get_connection() as connection:
        connection.execute("DELETE FROM users WHERE lower(email) = ?", (normalized_email,))


def delete_user_by_id(user_id: str) -> bool:
    if not user_id:
        return False
    if using_supabase():
        existing = _sb_select("users", {"id": user_id})
        if not existing:
            return False
        _sb_delete("users", {"id": user_id})
        return True

    with get_connection() as connection:
        result = connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return result.rowcount > 0


def hydrate_quiz(row: sqlite3.Row | dict | None, connection: sqlite3.Connection | None = None) -> dict | None:
    if not row:
        return None
    if using_supabase():
        quiz = dict(row)
        questions = _sort_rows(_sb_select("questions", {"quiz_id": quiz["id"]}), "sort_order")
        hydrated_questions = []
        total_points = 0
        for question in questions:
            options = _sort_rows(_sb_select("question_options", {"question_id": question["id"]}), "sort_order")
            option_texts = [option["option_text"] for option in options]
            correct_answer = next((option["option_text"] for option in options if option.get("is_correct")), "")
            hydrated_questions.append(
                {
                    "id": question["id"],
                    "question_text": question["question_text"],
                    "question_type": question["question_type"],
                    "options": option_texts,
                    "correct_answer": correct_answer,
                    "points": question["points"],
                }
            )
            total_points += int(question["points"])

        quiz["questions"] = hydrated_questions
        quiz["total_points"] = total_points
        return quiz

    quiz_id = row["id"]
    question_rows = connection.execute(
        """
        SELECT id, question_text, question_type, points
        FROM questions
        WHERE quiz_id = ?
        ORDER BY rowid
        """,
        (quiz_id,),
    ).fetchall()

    questions = []
    total_points = 0
    for question in question_rows:
        option_rows = connection.execute(
            """
            SELECT option_text, is_correct
            FROM question_options
            WHERE question_id = ?
            ORDER BY rowid
            """,
            (question["id"],),
        ).fetchall()
        options = [option["option_text"] for option in option_rows]
        correct_answer = next((option["option_text"] for option in option_rows if option["is_correct"]), "")
        questions.append(
            {
                "id": question["id"],
                "question_text": question["question_text"],
                "question_type": question["question_type"],
                "options": options,
                "correct_answer": correct_answer,
                "points": question["points"],
            }
        )
        total_points += question["points"]

    quiz = dict(row)
    quiz["questions"] = questions
    quiz["total_points"] = total_points
    return quiz


def get_quiz(quiz_id: str) -> dict | None:
    sync_quiz_statuses()
    if using_supabase():
        rows = _sb_select("quizzes", {"id": quiz_id})
        return hydrate_quiz(rows[0]) if rows else None

    with get_connection() as connection:
        row = connection.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        return hydrate_quiz(row, connection)


def get_quiz_by_code(code: str) -> dict | None:
    sync_quiz_statuses()
    if using_supabase():
        normalized_code = code.strip().upper()
        rows = [row for row in _sb_select("quizzes") if row.get("quiz_code", "").strip().upper() == normalized_code]
        return hydrate_quiz(rows[0]) if rows else None

    with get_connection() as connection:
        row = connection.execute("SELECT * FROM quizzes WHERE upper(quiz_code) = ?", (code.strip().upper(),)).fetchone()
        return hydrate_quiz(row, connection)


def get_quiz_attempt_for_student_code(quiz_id: str, student_id: str, quiz_code: str) -> dict | None:
    if using_supabase():
        rows = [row for row in _sb_select("quiz_attempts", {"quiz_id": quiz_id, "student_id": student_id}) if row.get("quiz_code", "") == quiz_code]
        return attempt_with_details(rows[0]) if rows else None

    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM quiz_attempts WHERE quiz_id = ? AND student_id = ? AND quiz_code = ? ORDER BY datetime(started_at) DESC, rowid DESC LIMIT 1",
            (quiz_id, student_id, quiz_code),
        ).fetchone()
        return attempt_with_details(row, connection) if row else None


def get_quiz_attempt_for_student(quiz_id: str, student_id: str) -> dict | None:
    if using_supabase():
        rows = _sort_rows(_sb_select("quiz_attempts", {"quiz_id": quiz_id, "student_id": student_id}), "started_at", reverse=True)
        return attempt_with_details(rows[0]) if rows else None

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM quiz_attempts
            WHERE quiz_id = ? AND student_id = ?
            ORDER BY datetime(started_at) DESC, rowid DESC
            LIMIT 1
            """,
            (quiz_id, student_id),
        ).fetchone()
        return attempt_with_details(row, connection) if row else None


def get_quizzes() -> list[dict]:
    sync_quiz_statuses()
    if using_supabase():
        rows = _sort_rows(_sb_select("quizzes"), "created_at", reverse=True)
        return [hydrate_quiz(row) for row in rows]

    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM quizzes ORDER BY datetime(created_at) DESC, rowid DESC").fetchall()
        return [hydrate_quiz(row, connection) for row in rows]


def open_quizzes() -> list[dict]:
    return [quiz for quiz in get_quizzes() if quiz["status"] == "published" and schedule_status(quiz) == "open"]


def create_or_update_quiz(
    *,
    quiz_id: str | None,
    creator_id: str,
    title: str,
    description: str,
    subject: str,
    time_limit_minutes: int,
    quiz_code: str,
    monitoring_enabled: bool,
    scheduled_start: str,
    scheduled_end: str,
    status: str,
    questions_payload: list[dict],
    assigned_section: str = "",
) -> str:
    time_limit_minutes = max(0, int(time_limit_minutes or 0))
    clean_assigned_section = " ".join(str(assigned_section or "").split())

    if using_supabase():
        now = _now_stamp()
        existing = _sb_select("quizzes", {"id": quiz_id}) if quiz_id else []
        if existing:
            _sb_update(
                "quizzes",
                {
                    "title": title,
                    "description": description,
                    "subject": subject,
                    "time_limit_minutes": time_limit_minutes,
                    "status": status,
                    "quiz_code": quiz_code,
                    "monitoring_enabled": bool(monitoring_enabled),
                    "assigned_section": clean_assigned_section,
                    "scheduled_start": scheduled_start or None,
                    "scheduled_end": scheduled_end or None,
                },
                {"id": quiz_id},
            )
            question_rows = _sb_select("questions", {"quiz_id": quiz_id})
            question_ids = [row["id"] for row in question_rows]
            _sb_delete_many("question_options", "question_id", question_ids)
            _sb_delete("questions", {"quiz_id": quiz_id})
        else:
            quiz_id = next_id("quiz")
            _sb_insert(
                "quizzes",
                {
                    "id": quiz_id,
                    "creator_id": creator_id,
                    "title": title,
                    "description": description,
                    "subject": subject,
                    "time_limit_minutes": time_limit_minutes,
                    "status": status,
                    "quiz_code": quiz_code,
                    "monitoring_enabled": bool(monitoring_enabled),
                    "assigned_section": clean_assigned_section,
                    "scheduled_start": scheduled_start or None,
                    "scheduled_end": scheduled_end or None,
                    "created_at": now,
                },
            )

        for index, question in enumerate(questions_payload):
            question_text = question.get("question_text", "").strip()
            if not question_text:
                continue
            question_id = next_id("question")
            question_type = question.get("question_type", "multiple_choice")
            points = int(question.get("points") or 1)
            _sb_insert(
                "questions",
                {
                    "id": question_id,
                    "quiz_id": quiz_id,
                    "question_text": question_text,
                    "question_type": question_type,
                    "points": points,
                    "sort_order": index,
                    "created_at": now,
                },
            )

            correct_answer = question.get("correct_answer", "")
            for option_index, option_text in enumerate(question.get("options", [])):
                clean_option = option_text.strip()
                if not clean_option:
                    continue
                _sb_insert(
                    "question_options",
                    {
                        "id": next_id("option"),
                        "question_id": question_id,
                        "option_text": clean_option,
                        "is_correct": clean_option == correct_answer,
                        "sort_order": option_index,
                        "created_at": now,
                    },
                )

        return quiz_id

    with get_connection() as connection:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if quiz_id:
            existing = connection.execute("SELECT id FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        else:
            existing = None

        if existing:
            connection.execute(
                """
                UPDATE quizzes
                SET title = ?, description = ?, subject = ?, time_limit_minutes = ?, status = ?,
                    quiz_code = ?, monitoring_enabled = ?, assigned_section = ?, scheduled_start = ?, scheduled_end = ?
                WHERE id = ?
                """,
                (
                    title,
                    description,
                    subject,
                    time_limit_minutes,
                    status,
                    quiz_code,
                    int(monitoring_enabled),
                    clean_assigned_section,
                    scheduled_start or None,
                    scheduled_end or None,
                    quiz_id,
                ),
            )
            question_ids = [
                row["id"]
                for row in connection.execute("SELECT id FROM questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
            ]
            if question_ids:
                placeholders = ",".join("?" * len(question_ids))
                connection.execute(f"DELETE FROM question_options WHERE question_id IN ({placeholders})", question_ids)
            connection.execute("DELETE FROM questions WHERE quiz_id = ?", (quiz_id,))
        else:
            quiz_id = next_id("quiz")
            connection.execute(
                """
                INSERT INTO quizzes (
                    id, creator_id, title, description, subject, time_limit_minutes, status,
                    quiz_code, monitoring_enabled, assigned_section, scheduled_start, scheduled_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quiz_id,
                    creator_id,
                    title,
                    description,
                    subject,
                    time_limit_minutes,
                    status,
                    quiz_code,
                    int(monitoring_enabled),
                    clean_assigned_section,
                    scheduled_start or None,
                    scheduled_end or None,
                    now,
                ),
            )

        for question in questions_payload:
            question_text = question.get("question_text", "").strip()
            if not question_text:
                continue
            question_id = next_id("question")
            question_type = question.get("question_type", "multiple_choice")
            points = int(question.get("points") or 1)
            connection.execute(
                """
                INSERT INTO questions (id, quiz_id, question_text, question_type, points)
                VALUES (?, ?, ?, ?, ?)
                """,
                (question_id, quiz_id, question_text, question_type, points),
            )

            for option_text in question.get("options", []):
                clean_option = option_text.strip()
                if not clean_option:
                    continue
                connection.execute(
                    """
                    INSERT INTO question_options (id, question_id, option_text, is_correct)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        next_id("option"),
                        question_id,
                        clean_option,
                        int(clean_option == question.get("correct_answer", "")),
                    ),
                )

    return quiz_id


def set_quiz_status(quiz_id: str, status: str) -> None:
    if using_supabase():
        _sb_update("quizzes", {"status": status}, {"id": quiz_id})
        return

    with get_connection() as connection:
        connection.execute("UPDATE quizzes SET status = ? WHERE id = ?", (status, quiz_id))


def delete_quiz_by_id(quiz_id: str) -> None:
    if using_supabase():
        questions = _sb_select("questions", {"quiz_id": quiz_id})
        question_ids = [row["id"] for row in questions]
        attempts = _sb_select("quiz_attempts", {"quiz_id": quiz_id})
        attempt_ids = [row["id"] for row in attempts]
        _sb_delete_many("question_options", "question_id", question_ids)
        _sb_delete_many("student_responses", "question_id", question_ids)
        _sb_delete_many("activity_logs", "attempt_id", attempt_ids)
        _sb_delete_many("student_responses", "attempt_id", attempt_ids)
        _sb_delete("questions", {"quiz_id": quiz_id})
        _sb_delete("quiz_attempts", {"quiz_id": quiz_id})
        _sb_delete("quizzes", {"id": quiz_id})
        return

    with get_connection() as connection:
        question_ids = [
            row["id"]
            for row in connection.execute("SELECT id FROM questions WHERE quiz_id = ?", (quiz_id,)).fetchall()
        ]
        attempt_ids = [
            row["id"]
            for row in connection.execute("SELECT id FROM quiz_attempts WHERE quiz_id = ?", (quiz_id,)).fetchall()
        ]
        if question_ids:
            placeholders = ",".join("?" * len(question_ids))
            connection.execute(f"DELETE FROM question_options WHERE question_id IN ({placeholders})", question_ids)
            connection.execute(f"DELETE FROM student_responses WHERE question_id IN ({placeholders})", question_ids)
        if attempt_ids:
            placeholders = ",".join("?" * len(attempt_ids))
            connection.execute(f"DELETE FROM activity_logs WHERE attempt_id IN ({placeholders})", attempt_ids)
            connection.execute(f"DELETE FROM student_responses WHERE attempt_id IN ({placeholders})", attempt_ids)
        connection.execute("DELETE FROM questions WHERE quiz_id = ?", (quiz_id,))
        connection.execute("DELETE FROM quiz_attempts WHERE quiz_id = ?", (quiz_id,))
        connection.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))


def attempt_with_details(row: sqlite3.Row | dict, connection: sqlite3.Connection | None = None) -> dict:
    attempt = dict(row)
    if using_supabase():
        quiz = get_quiz(attempt["quiz_id"])
        user = get_user_by_id(attempt["student_id"])
        response_rows = _sort_rows(_sb_select("student_responses", {"attempt_id": attempt["id"]}), "created_at")
        return {
            **attempt,
            "student_name": user["full_name"] if user else "Unknown Student",
            "student_email": user["email"] if user else "",
            "total_points": quiz["total_points"] if quiz else 0,
            "answers": [
                {
                    "question_id": response["question_id"],
                    "selected_answer": response.get("selected_option") or response.get("text_response"),
                    "is_correct": bool(response.get("is_correct")),
                }
                for response in response_rows
            ],
        }

    quiz = hydrate_quiz(connection.execute("SELECT * FROM quizzes WHERE id = ?", (attempt["quiz_id"],)).fetchone(), connection)
    user = get_user_by_id(attempt["student_id"])
    response_rows = connection.execute(
        """
        SELECT question_id, selected_option, text_response, is_correct
        FROM student_responses
        WHERE attempt_id = ?
        ORDER BY rowid
        """,
        (attempt["id"],),
    ).fetchall()
    return {
        **attempt,
        "student_name": user["full_name"] if user else "Unknown Student",
        "student_email": user["email"] if user else "",
        "total_points": quiz["total_points"] if quiz else 0,
        "answers": [
            {
                "question_id": response["question_id"],
                "selected_answer": response["selected_option"] or response["text_response"],
                "is_correct": bool(response["is_correct"]),
            }
            for response in response_rows
        ],
    }


def quiz_attempts(quiz_id: str) -> list[dict]:
    if using_supabase():
        rows = _sort_rows(_sb_select("quiz_attempts", {"quiz_id": quiz_id}), "started_at", reverse=True)
        return [attempt_with_details(row) for row in rows]

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM quiz_attempts WHERE quiz_id = ? ORDER BY datetime(started_at) DESC, rowid DESC",
            (quiz_id,),
        ).fetchall()
        return [attempt_with_details(row, connection) for row in rows]


def student_attempts(student_email: str) -> list[dict]:
    user = get_user_by_email(student_email)
    if not user:
        return []
    if using_supabase():
        rows = _sort_rows(_sb_select("quiz_attempts", {"student_id": user["id"]}), "started_at", reverse=True)
        return [attempt_with_details(row) for row in rows]

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM quiz_attempts WHERE student_id = ? ORDER BY datetime(started_at) DESC, rowid DESC",
            (user["id"],),
        ).fetchall()
        return [attempt_with_details(row, connection) for row in rows]


def get_attempt(attempt_id: str) -> dict | None:
    if using_supabase():
        rows = _sb_select("quiz_attempts", {"id": attempt_id})
        return attempt_with_details(rows[0]) if rows else None

    with get_connection() as connection:
        row = connection.execute("SELECT * FROM quiz_attempts WHERE id = ?", (attempt_id,)).fetchone()
        return attempt_with_details(row, connection) if row else None


def get_in_progress_attempt(quiz_id: str, student_id: str, quiz_code: str) -> dict | None:
    if using_supabase():
        rows = [
            row
            for row in _sb_select("quiz_attempts", {"quiz_id": quiz_id, "student_id": student_id, "status": "in_progress"})
            if row.get("quiz_code", "") == quiz_code
        ]
        rows = _sort_rows(rows, "started_at", reverse=True)
        return rows[0] if rows else None

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM quiz_attempts
            WHERE quiz_id = ? AND student_id = ? AND quiz_code = ? AND status = 'in_progress'
            ORDER BY datetime(started_at) DESC, rowid DESC
            LIMIT 1
            """,
            (quiz_id, student_id, quiz_code),
        ).fetchone()
        return dict(row) if row else None


def ensure_quiz_attempt_in_progress(quiz_id: str, student_id: str, consent_given: bool = False) -> str:
    quiz = get_quiz(quiz_id)
    if not quiz:
        raise ValueError("Quiz not found.")

    active_attempt = get_in_progress_attempt(quiz_id, student_id, quiz["quiz_code"])
    if active_attempt:
        if consent_given and not active_attempt.get("consent_given"):
            if using_supabase():
                _sb_update("quiz_attempts", {"consent_given": True}, {"id": active_attempt["id"]})
            else:
                with get_connection() as connection:
                    connection.execute(
                        "UPDATE quiz_attempts SET consent_given = 1 WHERE id = ?",
                        (active_attempt["id"],),
                    )
        return active_attempt["id"]

    started_at = _now_stamp()
    attempt_id = next_id("attempt")

    if using_supabase():
        _sb_insert(
            "quiz_attempts",
            {
                "id": attempt_id,
                "quiz_id": quiz_id,
                "student_id": student_id,
                "quiz_code": quiz["quiz_code"],
                "score": 0,
                "percentage": 0,
                "status": "in_progress",
                "started_at": started_at,
                "submitted_at": None,
                "consent_given": bool(consent_given),
                "created_at": started_at,
            },
        )
        return attempt_id

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO quiz_attempts (id, quiz_id, student_id, quiz_code, score, percentage, status, started_at, submitted_at, consent_given)
            VALUES (?, ?, ?, ?, 0, 0, 'in_progress', ?, NULL, ?)
            """,
            (attempt_id, quiz_id, student_id, quiz["quiz_code"], started_at, int(consent_given)),
        )
    return attempt_id


def attempt_deadline(quiz: dict | None, attempt: dict | None) -> datetime | None:
    if not quiz or not attempt:
        return None

    started_at = parse_schedule(attempt.get("started_at"))
    scheduled_end = parse_schedule(quiz.get("scheduled_end"))
    time_limit_minutes = max(0, int(quiz.get("time_limit_minutes") or 0))
    relative_deadline = (
        started_at + timedelta(minutes=time_limit_minutes)
        if started_at and time_limit_minutes > 0
        else None
    )

    if relative_deadline and scheduled_end:
        return min(relative_deadline, scheduled_end)
    return relative_deadline or scheduled_end


def remaining_attempt_seconds(quiz: dict | None, attempt: dict | None, now: datetime | None = None) -> int | None:
    deadline = attempt_deadline(quiz, attempt)
    if not deadline:
        return None
    current_time = now or datetime.now()
    return max(0, int((deadline - current_time).total_seconds()))


def attempt_has_expired(quiz: dict | None, attempt: dict | None, now: datetime | None = None) -> bool:
    deadline = attempt_deadline(quiz, attempt)
    if not deadline:
        return False
    return (now or datetime.now()) >= deadline


def finalize_quiz_attempt(
    attempt_id: str,
    answers: dict[str, str],
    consent_given: bool,
    status: str = "submitted",
) -> str:
    attempt = get_attempt(attempt_id)
    if not attempt:
        raise ValueError("Quiz attempt not found.")
    if attempt.get("status") in {"submitted", "auto_submitted"}:
        return attempt_id

    quiz = get_quiz(attempt["quiz_id"])
    if not quiz:
        raise ValueError("Quiz not found.")

    final_status = str(status or "submitted").strip().lower()
    if final_status not in {"submitted", "auto_submitted"}:
        final_status = "submitted"

    submitted_at = _now_stamp()
    score = 0
    total_points = quiz["total_points"]

    if using_supabase():
        _sb_delete("student_responses", {"attempt_id": attempt_id})
        for question in quiz["questions"]:
            selected_answer = answers.get(question["id"], "")
            is_correct = selected_answer == question["correct_answer"]
            if is_correct:
                score += question["points"]
            _sb_insert(
                "student_responses",
                {
                    "id": next_id("response"),
                    "attempt_id": attempt_id,
                    "question_id": question["id"],
                    "selected_option": selected_answer,
                    "text_response": "",
                    "is_correct": bool(is_correct),
                    "created_at": submitted_at,
                },
            )

        percentage = round((score / total_points) * 100, 2) if total_points else 0
        _sb_update(
            "quiz_attempts",
            {
                "score": score,
                "percentage": percentage,
                "status": final_status,
                "submitted_at": submitted_at,
                "consent_given": bool(consent_given),
            },
            {"id": attempt_id},
        )
        return attempt_id

    with get_connection() as connection:
        connection.execute("DELETE FROM student_responses WHERE attempt_id = ?", (attempt_id,))
        for question in quiz["questions"]:
            selected_answer = answers.get(question["id"], "")
            is_correct = selected_answer == question["correct_answer"]
            if is_correct:
                score += question["points"]
            connection.execute(
                """
                INSERT INTO student_responses (id, attempt_id, question_id, selected_option, text_response, is_correct)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    next_id("response"),
                    attempt_id,
                    question["id"],
                    selected_answer,
                    "",
                    int(is_correct),
                ),
            )

        percentage = round((score / total_points) * 100, 2) if total_points else 0
        connection.execute(
            """
            UPDATE quiz_attempts
            SET score = ?, percentage = ?, status = ?, submitted_at = ?, consent_given = ?
            WHERE id = ?
            """,
            (score, percentage, final_status, submitted_at, int(consent_given), attempt_id),
        )

    return attempt_id


def submit_quiz_attempt(quiz_id: str, student_id: str, answers: dict[str, str], consent_given: bool) -> str:
    attempt_id = ensure_quiz_attempt_in_progress(quiz_id, student_id, consent_given)
    return finalize_quiz_attempt(attempt_id, answers, consent_given)


def create_activity_log(
    quiz_id: str,
    attempt_id: str,
    event_type: str,
    event_description: str,
    flag_level: str = "low",
) -> dict:
    created_date = _now_stamp()
    payload = {
        "id": next_id("flag"),
        "quiz_id": quiz_id,
        "attempt_id": attempt_id,
        "event_type": event_type,
        "event_description": event_description,
        "flag_level": flag_level,
        "reviewed": False,
        "instructor_notes": "",
        "created_date": created_date,
    }

    if using_supabase():
        _sb_insert("activity_logs", payload)
        return payload

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO activity_logs (id, quiz_id, attempt_id, event_type, event_description, flag_level, reviewed, instructor_notes, created_date)
            VALUES (?, ?, ?, ?, ?, ?, 0, '', ?)
            """,
            (
                payload["id"],
                payload["quiz_id"],
                payload["attempt_id"],
                payload["event_type"],
                payload["event_description"],
                payload["flag_level"],
                payload["created_date"],
            ),
        )
    return payload


def activity_log_with_details(row: sqlite3.Row | dict) -> dict:
    flag = dict(row)
    if using_supabase():
        attempt = get_attempt(flag["attempt_id"]) if flag.get("attempt_id") else None
        return {
            **flag,
            "student_name": attempt["student_name"] if attempt else "Unknown Student",
            "student_email": attempt["student_email"] if attempt else "",
            "timestamp": flag.get("created_date") or flag.get("created_at"),
        }

    attempt = get_attempt(flag["attempt_id"]) if flag.get("attempt_id") else None
    return {
        **flag,
        "student_name": attempt["student_name"] if attempt else "Unknown Student",
        "student_email": attempt["student_email"] if attempt else "",
        "timestamp": flag["created_date"],
    }


def quiz_flags(quiz_id: str) -> list[dict]:
    if using_supabase():
        rows = _sort_rows(_sb_select("activity_logs", {"quiz_id": quiz_id}), "created_date", reverse=True)
    else:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, quiz_id, attempt_id, event_type, event_description, flag_level, reviewed, created_date
                FROM activity_logs
                WHERE quiz_id = ?
                ORDER BY datetime(created_date) DESC, rowid DESC
                """,
                (quiz_id,),
            ).fetchall()
    return [activity_log_with_details(row) for row in rows]


def sync_quiz_statuses(now: datetime | None = None) -> list[str]:
    current_time = now or datetime.now()
    closed_ids: list[str] = []

    if using_supabase():
        quizzes = _sb_select("quizzes")
        for quiz in quizzes:
            if quiz.get("status") != "published":
                continue
            end = parse_schedule(quiz.get("scheduled_end"))
            if end and current_time >= end:
                _sb_update("quizzes", {"status": "closed"}, {"id": quiz["id"]})
                closed_ids.append(str(quiz["id"]))
        return closed_ids

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, scheduled_end
            FROM quizzes
            WHERE status = 'published' AND scheduled_end IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            end = parse_schedule(row["scheduled_end"])
            if end and current_time >= end:
                connection.execute("UPDATE quizzes SET status = 'closed' WHERE id = ?", (row["id"],))
                closed_ids.append(str(row["id"]))
    return closed_ids

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM activity_logs WHERE quiz_id = ? ORDER BY datetime(created_date) DESC, rowid DESC",
            (quiz_id,),
        ).fetchall()
        return [activity_log_with_details(row) for row in rows]


def dashboard_stats() -> dict:
    if using_supabase():
        quizzes = get_quizzes()
        attempts = _sb_select("quiz_attempts")
        flags = _sb_select("activity_logs")
        percentages = [attempt.get("percentage", 0) for attempt in attempts if attempt.get("status") != "in_progress"]
        return {
            "total_quizzes": len(quizzes),
            "active_quizzes": sum(1 for quiz in quizzes if quiz["status"] == "published" and schedule_status(quiz) == "open"),
            "total_submissions": len([attempt for attempt in attempts if attempt.get("status") != "in_progress"]),
            "unreviewed_flags": sum(1 for flag in flags if not flag.get("reviewed")),
            "average_score": round(sum(percentages) / len(percentages), 1) if percentages else 0,
        }

    quizzes = get_quizzes()
    with get_connection() as connection:
        submission_rows = connection.execute(
            "SELECT percentage FROM quiz_attempts WHERE status != 'in_progress'"
        ).fetchall()
        unreviewed = connection.execute("SELECT COUNT(*) AS total FROM activity_logs WHERE reviewed = 0").fetchone()["total"]
    percentages = [row["percentage"] for row in submission_rows]
    return {
        "total_quizzes": len(quizzes),
        "active_quizzes": sum(1 for quiz in quizzes if quiz["status"] == "published" and schedule_status(quiz) == "open"),
        "total_submissions": len(submission_rows),
        "unreviewed_flags": unreviewed,
        "average_score": round(sum(percentages) / len(percentages), 1) if percentages else 0,
    }


def cheating_summary(quiz_id: str) -> dict:
    quiz = get_quiz(quiz_id)
    if not quiz:
        return {
            "quiz": None,
            "risk_level": "Low",
            "patterns": ["No suspicious pattern found in the selected quiz logs."],
            "students": ["No flagged students"],
            "overview": "No suspicious incidents were detected for this quiz based on the current monitoring logs.",
            "recommendation": "No immediate intervention is needed. Keep standard post-quiz review in place.",
            "flags_count": 0,
            "attempts_count": 0,
        }

    flags = quiz_flags(quiz["id"])
    attempts = quiz_attempts(quiz["id"])
    event_counter = Counter(flag["event_type"] for flag in flags)
    student_counter = Counter(flag["student_name"] for flag in flags)

    risk_score = sum({"low": 1, "medium": 2, "high": 3}.get(flag["flag_level"], 1) for flag in flags)
    risk = "Low"
    if risk_score >= 5 or any(flag["flag_level"] == "high" for flag in flags):
        risk = "High"
    elif risk_score >= 2:
        risk = "Medium"

    students = [name for name, _ in student_counter.most_common(3)] or ["No flagged students"]
    pattern_messages = {
        "alt_tab_attempt": "Alt+Tab attempts suggest the student tried to leave the quiz to look for outside answers.",
        "tab_switch": "Repeated tab switching suggests possible use of outside resources.",
        "window_blur": "Window blur events indicate the student left the exam screen during the attempt.",
        "copy_paste": "Copy or paste attempts suggest content transfer during the assessment.",
        "paste_attempt": "Paste attempts suggest prepared text may have been brought into the exam.",
        "right_click": "Right-click attempts suggest attempts to access browser tools or shortcuts.",
        "fullscreen_exit": "Fullscreen exits indicate the exam environment was intentionally interrupted.",
    }
    patterns = [pattern_messages.get(event_type, event_type.replace("_", " ").title()) for event_type, _ in event_counter.most_common(3)]
    if not patterns:
        patterns.append("No suspicious pattern found in the selected quiz logs.")

    if not flags:
        overview = "No suspicious incidents were detected for this quiz based on the current monitoring logs."
        recommendation = "No immediate intervention is needed. Keep standard post-quiz review in place."
    elif risk == "High":
        overview = f"This quiz shows a concentrated set of suspicious signals across {len(flags)} flagged events."
        recommendation = "Prioritize manual review of the flagged attempt before final grading."
    elif risk == "Medium":
        overview = f"This quiz shows moderate integrity concerns with {len(flags)} flagged events."
        recommendation = "Review flagged attempts first and compare suspicious timestamps with answer changes."
    else:
        overview = f"The selected quiz has limited monitoring concerns across {len(flags)} flagged events."
        recommendation = "Document the events, then proceed with normal review if no stronger evidence appears."

    return {
        "quiz": quiz,
        "risk_level": risk,
        "patterns": patterns,
        "students": students,
        "overview": overview,
        "recommendation": recommendation,
        "flags_count": len(flags),
        "attempts_count": len(attempts),
    }


def activity_stats() -> dict:
    if using_supabase():
        rows = _sb_select("activity_logs")
        return {
            "total_flags": len(rows),
            "high_severity": sum(1 for row in rows if row.get("flag_level") == "high"),
            "pending_review": sum(1 for row in rows if not row.get("reviewed")),
            "reviewed": sum(1 for row in rows if row.get("reviewed")),
        }

    with get_connection() as connection:
        rows = connection.execute("SELECT flag_level, reviewed FROM activity_logs").fetchall()
    return {
        "total_flags": len(rows),
        "high_severity": sum(1 for row in rows if row["flag_level"] == "high"),
        "pending_review": sum(1 for row in rows if not row["reviewed"]),
        "reviewed": sum(1 for row in rows if row["reviewed"]),
    }


def student_dashboard_summary(student_email: str) -> dict:
    attempts = student_attempts(student_email)
    completed = [attempt for attempt in attempts if attempt["status"] in {"submitted", "auto_submitted"}]
    percentages = [attempt["percentage"] for attempt in completed]
    average_score = round(sum(percentages) / len(percentages), 1) if percentages else 0
    best_score = max(percentages, default=0)
    passed = sum(1 for attempt in completed if attempt["percentage"] >= 75)
    performance = []
    for index, attempt in enumerate(completed[-4:], start=1):
        quiz = get_quiz(attempt["quiz_id"])
        performance.append(
            {
                "label": quiz["title"] if quiz else f"Quiz {index}",
                "short_label": f"Quiz {index}",
                "percentage": attempt["percentage"],
            }
        )

    history = []
    for attempt in completed:
        quiz = get_quiz(attempt["quiz_id"])
        history.append(
            {
                "title": quiz["title"] if quiz else attempt["quiz_id"],
                "subject": quiz["subject"] if quiz else "General",
                "date": attempt["submitted_at"].split(" ")[0] if attempt["submitted_at"] else attempt["started_at"].split(" ")[0],
                "percentage": attempt["percentage"],
                "score_text": f"{attempt['score']}/{attempt['total_points']}",
                "status": "Done" if attempt["status"] == "submitted" else "Auto Submitted",
            }
        )
    history.sort(key=lambda item: item["date"], reverse=True)
    return {
        "quizzes_taken": len(completed),
        "average_score": average_score,
        "best_score": best_score,
        "passed_count": passed,
        "pass_rate": round((passed / len(completed)) * 100, 1) if completed else 0,
        "performance": performance,
        "history": history,
    }


def schedule_status(quiz: dict, now: datetime | None = None) -> str:
    now = now or datetime.now()
    start = parse_schedule(quiz.get("scheduled_start"))
    end = parse_schedule(quiz.get("scheduled_end"))
    if start and now < start:
        return "upcoming"
    if end and now >= end:
        return "closed"
    return "open"


def quiz_access_state(quiz: dict, student_id: str | None = None, now: datetime | None = None) -> tuple[bool, str | None]:
    now = now or datetime.now()
    if quiz.get("status") != "published":
        return False, "This quiz is currently closed. Ask your instructor to reopen it."
    start = parse_schedule(quiz.get("scheduled_start"))
    end = parse_schedule(quiz.get("scheduled_end"))
    if start and now < start:
        return False, f"This quiz is scheduled to open on {format_schedule(quiz.get('scheduled_start'))}."
    if end and now >= end:
        return False, f"This quiz closed on {format_schedule(quiz.get('scheduled_end'))} and is now locked."
    if student_id:
        existing_attempt = get_quiz_attempt_for_student(quiz["id"], student_id)
        if existing_attempt:
            if existing_attempt.get("status") in {"submitted", "auto_submitted"}:
                return False, "You have already completed this quiz."
            if existing_attempt.get("status") == "in_progress" and attempt_has_expired(quiz, existing_attempt, now):
                return False, "Your attempt time has ended and the quiz is now locked."
            if existing_attempt.get("status") == "in_progress":
                return True, None
        assigned_section = " ".join(str(quiz.get("assigned_section", "")).split())
        if assigned_section:
            student = get_user_by_id(student_id)
            student_section = " ".join(str((student or {}).get("section_name", "")).split())
            if _normalize_section_name(student_section) != _normalize_section_name(assigned_section):
                if student_section:
                    return False, f"This quiz is only available to {assigned_section}. Your account is enrolled in {student_section}."
                return False, f"This quiz is only available to {assigned_section}. Your account does not have a section or class assigned yet."
    return True, None


def reset_dashboard_data() -> dict[str, int]:
    if using_supabase():
        attempts = _sb_select("quiz_attempts")
        attempt_ids = [str(attempt.get("id", "")).strip() for attempt in attempts if str(attempt.get("id", "")).strip()]
        responses = _sb_select("student_responses")
        flags = _sb_select("activity_logs")
        _sb_delete_many("student_responses", "attempt_id", attempt_ids)
        for flag in flags:
            flag_id = str(flag.get("id", "")).strip()
            if flag_id:
                _sb_delete("activity_logs", {"id": flag_id})
        for attempt_id in attempt_ids:
            _sb_delete("quiz_attempts", {"id": attempt_id})
        return {
            "attempts": len(attempt_ids),
            "responses": len(responses),
            "activity_logs": len(flags),
        }

    with get_connection() as connection:
        attempt_count = connection.execute("SELECT COUNT(*) AS total FROM quiz_attempts").fetchone()["total"]
        response_count = connection.execute("SELECT COUNT(*) AS total FROM student_responses").fetchone()["total"]
        flag_count = connection.execute("SELECT COUNT(*) AS total FROM activity_logs").fetchone()["total"]
        connection.execute("DELETE FROM activity_logs")
        connection.execute("DELETE FROM student_responses")
        connection.execute("DELETE FROM quiz_attempts")

    return {
        "attempts": int(attempt_count),
        "responses": int(response_count),
        "activity_logs": int(flag_count),
    }


def _schedule_day_span(quiz: dict) -> tuple[date, date] | None:
    start = parse_schedule(quiz.get("scheduled_start"))
    end = parse_schedule(quiz.get("scheduled_end"))
    if not start and not end:
        return None

    start_day = (start or end).date()
    end_day = (end or start).date()
    if end_day < start_day:
        start_day, end_day = end_day, start_day
    return start_day, end_day


def _sort_scheduled_quizzes(quizzes: list[dict]) -> list[dict]:
    return sorted(
        quizzes,
        key=lambda quiz: (
            parse_schedule(quiz.get("scheduled_start"))
            or parse_schedule(quiz.get("scheduled_end"))
            or datetime.max,
            str(quiz.get("title", "")).lower(),
        ),
    )


def scheduled_quizzes_for_day(day_value: str | date | datetime | None) -> list[dict]:
    if not day_value:
        return []

    if isinstance(day_value, datetime):
        target_day = day_value.date()
    elif isinstance(day_value, date):
        target_day = day_value
    else:
        try:
            target_day = datetime.strptime(str(day_value).strip(), "%Y-%m-%d").date()
        except ValueError:
            return []

    scheduled = []
    for quiz in get_quizzes():
        span = _schedule_day_span(quiz)
        if not span:
            continue
        start_day, end_day = span
        if start_day <= target_day <= end_day:
            scheduled.append(quiz)

    return _sort_scheduled_quizzes(scheduled)


def scheduled_quizzes_by_day(year: int, month: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)

    for quiz in get_quizzes():
        span = _schedule_day_span(quiz)
        if not span:
            continue
        start_day, end_day = span
        range_start = max(start_day, month_start)
        range_end = min(end_day, month_end)
        if range_end < range_start:
            continue

        current_day = range_start
        while current_day <= range_end:
            key = current_day.strftime("%Y-%m-%d")
            grouped.setdefault(key, []).append(quiz)
            current_day += timedelta(days=1)

    for key, quizzes in grouped.items():
        grouped[key] = _sort_scheduled_quizzes(quizzes)
    return grouped


def build_dashboard_calendar(year: int, month: int) -> list[list[dict]]:
    month_grid = calendar.Calendar(firstweekday=6).monthdatescalendar(year, month)
    grouped = scheduled_quizzes_by_day(year, month)
    rows: list[list[dict]] = []
    for week in month_grid:
        row = []
        for day in week:
            day_key = day.strftime("%Y-%m-%d")
            row.append(
                {
                    "date": day,
                    "key": day_key,
                    "in_month": day.month == month,
                    "quizzes": grouped.get(day_key, []),
                }
            )
        rows.append(row)
    return rows
