from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env_loader import load_env_file

load_env_file(ROOT_DIR / ".env")

DEFAULT_SOURCE_DB = Path(
    os.environ.get("DB_PATH", str(ROOT_DIR / "database" / "semcds.db"))
)
DEFAULT_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

TABLE_ORDER = [
    "users",
    "quizzes",
    "questions",
    "question_options",
    "quiz_attempts",
    "student_responses",
    "activity_logs",
]

TABLE_COLUMNS = {
    "users": [
        "id",
        "email",
        "full_name",
        "role",
        "section_name",
        "avatar_url",
        "password_hash",
        "created_at",
    ],
    "quizzes": [
        "id",
        "creator_id",
        "title",
        "description",
        "subject",
        "time_limit_minutes",
        "status",
        "quiz_code",
        "monitoring_enabled",
        "assigned_section",
        "scheduled_start",
        "scheduled_end",
        "created_at",
    ],
    "questions": [
        "id",
        "quiz_id",
        "question_text",
        "question_type",
        "points",
        "sort_order",
        "created_at",
    ],
    "question_options": [
        "id",
        "question_id",
        "option_text",
        "is_correct",
        "sort_order",
        "created_at",
    ],
    "quiz_attempts": [
        "id",
        "quiz_id",
        "student_id",
        "quiz_code",
        "score",
        "percentage",
        "status",
        "started_at",
        "submitted_at",
        "consent_given",
        "created_at",
    ],
    "student_responses": [
        "id",
        "attempt_id",
        "question_id",
        "selected_option",
        "text_response",
        "is_correct",
        "created_at",
    ],
    "activity_logs": [
        "id",
        "quiz_id",
        "attempt_id",
        "event_type",
        "event_description",
        "flag_level",
        "reviewed",
        "instructor_notes",
        "created_date",
    ],
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clean_text(value, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _clean_optional_text(value):
    if value in {None, ""}:
        return None
    return str(value)


def _chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


class SupabaseRestClient:
    def __init__(self, base_url: str, service_role_key: str, timeout_seconds: int = 60):
        self.base_url = base_url.rstrip("/") + "/rest/v1"
        self.service_role_key = service_role_key
        self.timeout_seconds = timeout_seconds

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _query_string(self, params: dict[str, str] | None) -> str:
        if not params:
            return ""
        return "&".join(
            f"{urllib_parse.quote(str(key))}={urllib_parse.quote(str(value), safe='.,()*')}"
            for key, value in params.items()
        )

    def request(
        self,
        method: str,
        table_name: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict | list[dict] | None = None,
        prefer: str | None = None,
    ) -> list[dict]:
        url = f"{self.base_url}/{table_name}"
        query = self._query_string(params)
        if query:
            url = f"{url}?{query}"

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib_request.Request(
            url,
            data=data,
            headers=self._headers(prefer),
            method=method,
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise RuntimeError(f"{method} {table_name} failed: {error_body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Unable to connect to Supabase: {exc}") from exc

        if not body.strip():
            return []
        parsed = json.loads(body)
        return parsed if isinstance(parsed, list) else [parsed]

    def validate_table_shape(self, table_name: str, columns: list[str]) -> None:
        self.request(
            "GET",
            table_name,
            params={
                "select": ",".join(columns),
                "limit": "1",
            },
        )

    def delete_all(self, table_name: str) -> None:
        self.request(
            "DELETE",
            table_name,
            params={"id": "not.is.null"},
            prefer="return=minimal",
        )

    def upsert_rows(self, table_name: str, rows: list[dict], batch_size: int) -> None:
        if not rows:
            return
        for chunk in _chunked(rows, batch_size):
            self.request(
                "POST",
                table_name,
                params={"on_conflict": "id"},
                payload=chunk,
                prefer="resolution=merge-duplicates,return=minimal",
            )


def _get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def load_sqlite_payloads(db_path: Path) -> dict[str, list[dict]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        user_columns = _get_table_columns(connection, "users")
        quiz_columns = _get_table_columns(connection, "quizzes")
        question_columns = _get_table_columns(connection, "questions")
        option_columns = _get_table_columns(connection, "question_options")
        attempt_columns = _get_table_columns(connection, "quiz_attempts")
        response_columns = _get_table_columns(connection, "student_responses")
        activity_columns = _get_table_columns(connection, "activity_logs")

        users = []
        for row in connection.execute(
            "SELECT rowid, * FROM users ORDER BY datetime(created_at), rowid"
        ).fetchall():
            users.append(
                {
                    "id": _clean_text(row["id"]),
                    "email": _clean_text(row["email"]).strip().lower(),
                    "full_name": _clean_text(row["full_name"]).strip(),
                    "role": _clean_text(row["role"]).strip().lower(),
                    "section_name": _clean_text(row["section_name"]).strip() if "section_name" in user_columns else "",
                    "avatar_url": _clean_text(row["avatar_url"]).strip() if "avatar_url" in user_columns else "",
                    "password_hash": _clean_text(row["password_hash"]),
                    "created_at": _clean_text(row["created_at"], _now_stamp()),
                }
            )

        quizzes = []
        quiz_created_at: dict[str, str] = {}
        for row in connection.execute(
            "SELECT rowid, * FROM quizzes ORDER BY datetime(created_at), rowid"
        ).fetchall():
            created_at = _clean_text(row["created_at"], _now_stamp())
            quiz_id = _clean_text(row["id"])
            quiz_created_at[quiz_id] = created_at
            quizzes.append(
                {
                    "id": quiz_id,
                    "creator_id": _clean_text(row["creator_id"]),
                    "title": _clean_text(row["title"]).strip(),
                    "description": _clean_optional_text(row["description"]),
                    "subject": _clean_text(row["subject"]).strip(),
                    "time_limit_minutes": int(row["time_limit_minutes"] or 0),
                    "status": _clean_text(row["status"]).strip().lower(),
                    "quiz_code": _clean_text(row["quiz_code"]).strip(),
                    "monitoring_enabled": _normalize_bool(row["monitoring_enabled"]),
                    "assigned_section": _clean_text(row["assigned_section"]).strip() if "assigned_section" in quiz_columns else "",
                    "scheduled_start": _clean_optional_text(row["scheduled_start"]),
                    "scheduled_end": _clean_optional_text(row["scheduled_end"]),
                    "created_at": created_at,
                }
            )

        questions = []
        question_created_at: dict[str, str] = {}
        question_position: defaultdict[str, int] = defaultdict(int)
        for row in connection.execute(
            "SELECT rowid, * FROM questions ORDER BY quiz_id, rowid"
        ).fetchall():
            quiz_id = _clean_text(row["quiz_id"])
            sort_order = int(row["sort_order"]) if "sort_order" in question_columns else question_position[quiz_id]
            question_position[quiz_id] += 1
            created_at = (
                _clean_text(row["created_at"], "")
                if "created_at" in question_columns
                else ""
            ) or quiz_created_at.get(quiz_id, _now_stamp())
            question_id = _clean_text(row["id"])
            question_created_at[question_id] = created_at
            questions.append(
                {
                    "id": question_id,
                    "quiz_id": quiz_id,
                    "question_text": _clean_text(row["question_text"]).strip(),
                    "question_type": _clean_text(row["question_type"]).strip(),
                    "points": int(row["points"] or 1),
                    "sort_order": sort_order,
                    "created_at": created_at,
                }
            )

        question_options = []
        option_position: defaultdict[str, int] = defaultdict(int)
        for row in connection.execute(
            "SELECT rowid, * FROM question_options ORDER BY question_id, rowid"
        ).fetchall():
            question_id = _clean_text(row["question_id"])
            sort_order = int(row["sort_order"]) if "sort_order" in option_columns else option_position[question_id]
            option_position[question_id] += 1
            created_at = (
                _clean_text(row["created_at"], "")
                if "created_at" in option_columns
                else ""
            ) or question_created_at.get(question_id, _now_stamp())
            question_options.append(
                {
                    "id": _clean_text(row["id"]),
                    "question_id": question_id,
                    "option_text": _clean_text(row["option_text"]).strip(),
                    "is_correct": _normalize_bool(row["is_correct"]),
                    "sort_order": sort_order,
                    "created_at": created_at,
                }
            )

        quiz_attempts = []
        attempt_created_at: dict[str, str] = {}
        for row in connection.execute(
            "SELECT rowid, * FROM quiz_attempts ORDER BY datetime(started_at), rowid"
        ).fetchall():
            started_at = _clean_text(row["started_at"], _now_stamp())
            created_at = (
                _clean_text(row["created_at"], "")
                if "created_at" in attempt_columns
                else ""
            ) or started_at
            attempt_id = _clean_text(row["id"])
            attempt_created_at[attempt_id] = _clean_text(row["submitted_at"], "") or started_at
            quiz_attempts.append(
                {
                    "id": attempt_id,
                    "quiz_id": _clean_text(row["quiz_id"]),
                    "student_id": _clean_text(row["student_id"]),
                    "quiz_code": _clean_text(row["quiz_code"]).strip(),
                    "score": int(row["score"] or 0),
                    "percentage": float(row["percentage"] or 0),
                    "status": _clean_text(row["status"]).strip().lower(),
                    "started_at": started_at,
                    "submitted_at": _clean_optional_text(row["submitted_at"]),
                    "consent_given": _normalize_bool(row["consent_given"]),
                    "created_at": created_at,
                }
            )

        student_responses = []
        for row in connection.execute(
            "SELECT rowid, * FROM student_responses ORDER BY attempt_id, rowid"
        ).fetchall():
            attempt_id = _clean_text(row["attempt_id"])
            created_at = (
                _clean_text(row["created_at"], "")
                if "created_at" in response_columns
                else ""
            ) or attempt_created_at.get(attempt_id, _now_stamp())
            student_responses.append(
                {
                    "id": _clean_text(row["id"]),
                    "attempt_id": attempt_id,
                    "question_id": _clean_text(row["question_id"]),
                    "selected_option": _clean_optional_text(row["selected_option"]),
                    "text_response": _clean_optional_text(row["text_response"]),
                    "is_correct": _normalize_bool(row["is_correct"]),
                    "created_at": created_at,
                }
            )

        activity_logs = []
        for row in connection.execute(
            "SELECT rowid, * FROM activity_logs ORDER BY datetime(created_date), rowid"
        ).fetchall():
            created_date = (
                _clean_text(row["created_date"], "")
                if "created_date" in activity_columns
                else ""
            ) or _now_stamp()
            activity_logs.append(
                {
                    "id": _clean_text(row["id"]),
                    "quiz_id": _clean_text(row["quiz_id"]),
                    "attempt_id": _clean_text(row["attempt_id"]),
                    "event_type": _clean_text(row["event_type"]).strip(),
                    "event_description": _clean_text(row["event_description"]).strip(),
                    "flag_level": _clean_text(row["flag_level"], "low").strip().lower(),
                    "reviewed": _normalize_bool(row["reviewed"]),
                    "instructor_notes": _clean_optional_text(row["instructor_notes"]) or "",
                    "created_date": created_date,
                }
            )

        return {
            "users": users,
            "quizzes": quizzes,
            "questions": questions,
            "question_options": question_options,
            "quiz_attempts": quiz_attempts,
            "student_responses": student_responses,
            "activity_logs": activity_logs,
        }
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate SEMCDS data from local SQLite to Supabase."
    )
    parser.add_argument(
        "--source-db",
        default=str(DEFAULT_SOURCE_DB),
        help="Path to the SQLite database file. Defaults to DB_PATH from .env.",
    )
    parser.add_argument(
        "--supabase-url",
        default=DEFAULT_SUPABASE_URL,
        help="Supabase project URL. Defaults to SUPABASE_URL from .env.",
    )
    parser.add_argument(
        "--service-role-key",
        default=DEFAULT_SUPABASE_SERVICE_ROLE_KEY,
        help="Supabase service role key. Defaults to SUPABASE_SERVICE_ROLE_KEY from .env.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of rows to upsert per REST request.",
    )
    parser.add_argument(
        "--truncate-first",
        action="store_true",
        help="Delete existing Supabase rows first, in reverse dependency order.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip schema validation. Use only if you already know the destination schema is correct.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_db = Path(args.source_db).expanduser()
    supabase_url = str(args.supabase_url or "").strip()
    service_role_key = str(args.service_role_key or "").strip()

    if not source_db.exists():
        print(f"SQLite database not found: {source_db}", file=sys.stderr)
        return 1
    if not supabase_url or not service_role_key:
        print(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env, or pass them as CLI flags.",
            file=sys.stderr,
        )
        return 1

    client = SupabaseRestClient(supabase_url, service_role_key)

    if not args.skip_preflight:
        try:
            for table_name in TABLE_ORDER:
                client.validate_table_shape(table_name, TABLE_COLUMNS[table_name])
        except RuntimeError as exc:
            print(
                "Supabase schema validation failed.\n"
                "Run database/supabase_schema.sql in the Supabase SQL Editor first.\n"
                f"Details: {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        payloads = load_sqlite_payloads(source_db)
    except Exception as exc:
        print(f"Failed to read SQLite source database: {exc}", file=sys.stderr)
        return 1

    print("Source row counts:")
    for table_name in TABLE_ORDER:
        print(f"  {table_name}: {len(payloads[table_name])}")

    try:
        if args.truncate_first:
            print("Clearing destination tables first...")
            for table_name in reversed(TABLE_ORDER):
                client.delete_all(table_name)

        for table_name in TABLE_ORDER:
            rows = payloads[table_name]
            if not rows:
                continue
            print(f"Migrating {table_name} ({len(rows)} rows)...")
            client.upsert_rows(table_name, rows, batch_size=max(1, int(args.batch_size)))
    except RuntimeError as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1

    print("Supabase migration completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
