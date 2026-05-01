from __future__ import annotations

from pathlib import Path

import base64
import csv
import hashlib
import json
import os
import re
import secrets
import sqlite3
import ssl
import smtplib
from email.message import EmailMessage
from io import BytesIO, StringIO
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pypdf import PdfReader

from .env_loader import load_env_file

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
except Exception:  # pragma: no cover - optional realtime dependency
    SocketIO = None
    emit = None
    join_room = None
    leave_room = None

load_env_file()

from .data import (
    activity_log_with_details,
    attempt_has_expired,
    build_dashboard_calendar,
    cheating_summary,
    create_activity_log,
    create_or_update_quiz,
    create_user,
    dashboard_stats,
    delete_quiz_by_id,
    delete_user_by_id,
    delete_user_by_email,
    ensure_quiz_attempt_in_progress,
    finalize_quiz_attempt,
    format_schedule,
    get_attempt,
    get_quiz,
    get_quiz_attempt_for_student,
    get_quiz_by_code,
    get_quizzes,
    get_user_by_email,
    get_user_by_id,
    get_users,
    init_database,
    open_quizzes,
    quiz_attempts,
    quiz_access_state,
    quiz_flags,
    parse_schedule,
    remaining_attempt_seconds,
    reset_dashboard_data,
    schedule_status,
    scheduled_quizzes_for_day,
    student_dashboard_summary,
    set_user_avatar_url,
    set_user_password,
    verify_password,
    set_quiz_status,
    sync_quiz_statuses,
    update_user,
    user_record_counts,
)


DETECTION_MODEL_PATH = Path(__file__).resolve().parent.parent / "best.pt"
MIN_CONFIDENCE_NORMAL = 0.45
MIN_CONFIDENCE_DETECTION = 0.20
MIN_FACE_AREA_RATIO = 0.05
MAX_FACE_AREA_RATIO = 0.60
CENTER_MARGIN_RATIO = 0.25
DETECTION_INFER_CONFIDENCE = 0.15
DETECTION_INFER_IOU = 0.45
DETECTION_INFER_IMGSZ = 416
DETECTION_INFER_MAX_DET = 3
DETECTION_CLASS_NORMAL_MIN_CONF = 0.30
DETECTION_CLASS_CHEAT_MIN_CONF = 0.55
DETECTION_CLASS_MARGIN = 0.06
DETECTION_CLASS_DRAW_MIN_CONF = 0.15
DETECTION_CLASS_CHEAT_STRICT_MIN_CONF = 0.70
QUIZ_SECTION_OPTIONS = ["BSIT-NT 3201", "BSIT-NT 3202"]
SOCKETIO_ASYNC_MODE = os.getenv("SOCKETIO_ASYNC_MODE", "threading").strip().lower()
if SOCKETIO_ASYNC_MODE in {"", "auto"}:
    SOCKETIO_ASYNC_MODE = None

socketio = SocketIO(async_mode=SOCKETIO_ASYNC_MODE) if SocketIO else None
_monitor_rooms: dict[str, dict[str, dict]] = {}
_detection_event_cache: dict[str, datetime] = {}

ACTIVITY_EVENT_LABELS = {
    "cheat": "Cheat Detected",
    "alt_tab_attempt": "Alt+Tab Attempt",
    "face_detected": "Face Detected",
    "no_face_detected": "No Face Detected",
    "low_confidence_face_detected": "Low Confidence Face Detected",
    "camera_permission_denied": "Camera Permission Denied",
    "camera_api_unavailable": "Camera API Unavailable",
    "camera_status": "Camera Status Issue",
    "tab_switch": "Tab Switch Detected",
    "window_blur": "Quiz Window Lost Focus",
    "detection_request_failed": "Detection Request Failed",
    "detection_unavailable": "Detection Unavailable",
    "realtime_error": "Realtime Error",
    "webrtc_connection_error": "Camera Stream Connection Error",
    "webrtc_stream_disconnected": "Camera Stream Disconnected",
}
ATTEMPT_STATUS_LABELS = {
    "submitted": "Submitted",
    "auto_submitted": "Auto Submitted",
    "in_progress": "In Progress",
}
ATTEMPT_STATUS_CLASSES = {
    "submitted": "submitted",
    "auto_submitted": "auto-submitted",
    "in_progress": "in-progress",
}


def humanize_identifier(value: str | None, fallback: str = "") -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return fallback
    normalized = raw_value.replace("-", "_").strip("_").lower()
    if normalized in ACTIVITY_EVENT_LABELS:
        return ACTIVITY_EVENT_LABELS[normalized]
    return normalized.replace("_", " ").title()


def activity_event_label(event_type: str | None) -> str:
    return humanize_identifier(event_type, "Activity Event")


def activity_result_label(event_type: str | None) -> str:
    normalized = str(event_type or "").strip().replace("-", "_").lower()
    if not normalized:
        return "Waiting for Signal"
    if normalized == "normal":
        return "Normal"
    return activity_event_label(normalized)


def attempt_status_label(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    return ATTEMPT_STATUS_LABELS.get(normalized, humanize_identifier(normalized, "Unknown"))


def attempt_status_class(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    return ATTEMPT_STATUS_CLASSES.get(normalized, "neutral")


def review_status_class(reviewed: bool) -> str:
    return "reviewed" if reviewed else "pending"


def build_activity_stats(logs: list[dict]) -> dict:
    return {
        "total_flags": len(logs),
        "high_severity": sum(1 for row in logs if row.get("flag_level") == "high"),
        "pending_review": sum(1 for row in logs if not row.get("reviewed")),
        "reviewed": sum(1 for row in logs if row.get("reviewed")),
    }


def sort_timestamp(value: str | None) -> datetime:
    return parse_schedule(value) or datetime.min


def format_current_timestamp(value: datetime) -> str:
    return value.strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")


def parse_dashboard_day_key(value: str | None) -> datetime | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d")
    except ValueError:
        return None


def format_dashboard_day_label(value: datetime | None) -> str:
    if not value:
        return ""
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def student_quiz_window_summary(quiz: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    start = parse_schedule(quiz.get("scheduled_start"))
    end = parse_schedule(quiz.get("scheduled_end"))
    if start and end:
        summary = f"Open now until {format_schedule(quiz.get('scheduled_end'))}."
    elif end:
        summary = f"Open now. Closes at {format_schedule(quiz.get('scheduled_end'))}."
    elif start:
        summary = f"Open now. Started at {format_schedule(quiz.get('scheduled_start'))}."
    else:
        summary = "Open now. No closing time is scheduled."
    return {
        "start": start,
        "end": end,
        "opens_at": format_schedule(quiz.get("scheduled_start")) if start else "Immediately when published",
        "closes_at": format_schedule(quiz.get("scheduled_end")) if end else "No closing time set",
        "summary": summary,
        "is_open_now": True,
        "checked_at": format_current_timestamp(now),
    }


def extract_checkpoint_labels(model_path: Path) -> list[str]:
    """Best-effort label extraction from a corrupted Ultralytics checkpoint."""
    if not model_path.exists():
        return []

    try:
        raw_bytes = model_path.read_bytes()
    except OSError:
        return []

    names_anchor = raw_bytes.find(b"names")
    if names_anchor < 0:
        return []

    window = raw_bytes[names_anchor:names_anchor + 512]
    labels: list[str] = []
    marker = b"X"
    cursor = 0

    while cursor < len(window):
        marker_index = window.find(marker, cursor)
        if marker_index < 0 or marker_index + 5 > len(window):
            break

        raw_length = window[marker_index + 1:marker_index + 5]
        text_length = int.from_bytes(raw_length, "little", signed=False)
        text_start = marker_index + 5
        text_end = text_start + text_length
        cursor = marker_index + 1

        if text_length <= 0 or text_end > len(window):
            continue

        candidate_bytes = window[text_start:text_end]
        if not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]{1,31}", candidate_bytes):
            continue

        label = candidate_bytes.decode("ascii", errors="ignore").strip().lower()
        if label in {
            "names",
            "save",
            "train",
            "model",
            "data",
            "detect",
            "task",
            "mode",
            "args",
            "epochs",
            "time",
            "patience",
            "batch",
            "imgsz",
        }:
            continue
        if label not in labels:
            labels.append(label)
    return labels


def classify_face_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    frame_width: int,
    frame_height: int,
) -> tuple[bool, str]:
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    box_area_ratio = box_area / frame_area if frame_area > 0 else 0

    if confidence < MIN_CONFIDENCE_DETECTION:
        return False, "very_low_confidence"
    if box_area_ratio < MIN_FACE_AREA_RATIO:
        return False, "face_too_small"
    if box_area_ratio > MAX_FACE_AREA_RATIO:
        return False, "face_too_large"

    face_center_x = (x1 + x2) / 2
    center_left = frame_width * CENTER_MARGIN_RATIO
    center_right = frame_width * (1 - CENTER_MARGIN_RATIO)
    if face_center_x < center_left or face_center_x > center_right:
        return False, "face_off_center"

    if confidence < MIN_CONFIDENCE_NORMAL:
        return False, "low_confidence"

    return True, ""


def create_app() -> Flask:
    app = Flask(__name__)
    configured_secret_key = os.environ.get("SECRET_KEY", "").strip() or os.environ.get("FLASK_SECRET_KEY", "").strip()
    fallback_secret_key = hashlib.sha256(str(Path(__file__).resolve().parent.parent).encode("utf-8")).hexdigest()
    app.config["SECRET_KEY"] = configured_secret_key or fallback_secret_key
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
    app.config["PREFERRED_URL_SCHEME"] = "https"  # Use https by default for ngrok
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
        "SESSION_COOKIE_SECURE",
        "1" if os.getenv("FLASK_ENV", "development") == "production" else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    app.config["SESSION_COOKIE_SAMESITE"] = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_NAME"] = "semcds_session"
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", str(15 * 1024 * 1024)))
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    init_database()

    if socketio:
        socketio.init_app(app, manage_session=False)

    def redirect_to(endpoint, **kwargs):
        """Redirect to an endpoint using relative URLs to preserve the current host.
        This ensures redirects work correctly whether accessed via localhost or ngrok URL."""
        return redirect(url_for(endpoint, **kwargs, _external=False))

    def _normalize_avatar_url(raw_value: str | None) -> str:
        avatar_url = str(raw_value or "").strip()
        if not avatar_url:
            return ""
        parsed = urllib_parse.urlparse(avatar_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return avatar_url

    def current_session_user() -> dict | None:
        user_id = str(session.get("user_id", "")).strip()
        role = str(session.get("role", "")).strip()
        if not user_id or role not in {"admin", "user"}:
            return None

        user = get_user_by_id(user_id)
        if not user or user.get("role") != role:
            session.clear()
            return None
        avatar_url = _normalize_avatar_url(session.get("avatar_url") or user.get("avatar_url"))
        return {**user, "avatar_url": avatar_url}

    def issue_csrf_token(force: bool = False) -> str:
        if force or not session.get("_csrf_token"):
            session["_csrf_token"] = secrets.token_urlsafe(32)
        return str(session["_csrf_token"])

    app.jinja_env.globals["csrf_token"] = issue_csrf_token
    app.jinja_env.globals["activity_event_label"] = activity_event_label
    app.jinja_env.globals["activity_result_label"] = activity_result_label
    app.jinja_env.globals["attempt_status_label"] = attempt_status_label
    app.jinja_env.globals["attempt_status_class"] = attempt_status_class
    app.jinja_env.globals["review_status_class"] = review_status_class

    def build_quiz_results_context(quiz: dict) -> dict:
        attempts = quiz_attempts(quiz["id"])
        flags = quiz_flags(quiz["id"])
        flags_by_attempt = Counter(
            str(flag.get("attempt_id", "")).strip()
            for flag in flags
            if str(flag.get("attempt_id", "")).strip()
        )
        completed_attempts = [
            attempt for attempt in attempts
            if attempt.get("status") in {"submitted", "auto_submitted"}
        ]
        ranked_attempts = sorted(
            completed_attempts,
            key=lambda item: (
                -float(item.get("percentage", 0) or 0),
                -int(item.get("score", 0) or 0),
                parse_schedule(item.get("submitted_at")) or datetime.max,
                item.get("student_name", "").lower(),
            ),
        )

        previous_score_key = None
        current_rank = 0
        for index, attempt in enumerate(ranked_attempts, start=1):
            score_key = (
                float(attempt.get("percentage", 0) or 0),
                int(attempt.get("score", 0) or 0),
            )
            if score_key != previous_score_key:
                current_rank = index
                previous_score_key = score_key
            attempt["rank"] = current_rank

        for attempt in attempts:
            if attempt.get("status") not in {"submitted", "auto_submitted"}:
                attempt["rank"] = None

        ordered_attempts = ranked_attempts + [
            attempt for attempt in attempts
            if attempt.get("status") not in {"submitted", "auto_submitted"}
        ]
        average = round(sum(item["percentage"] for item in ranked_attempts) / len(ranked_attempts), 1) if ranked_attempts else 0
        highest = max((item["percentage"] for item in ranked_attempts), default=0)
        return {
            "attempts": ordered_attempts,
            "all_flags": flags,
            "flags_by_attempt": flags_by_attempt,
            "top_rankings": ranked_attempts[:3],
            "metrics": {
                "submissions": len(ranked_attempts),
                "average_score": average,
                "highest_score": highest,
                "flags_count": len(flags),
            },
        }

    def quiz_results_csv_filename(quiz: dict) -> str:
        raw_title = str(quiz.get("title", "")).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw_title).strip("-")
        return f"{slug or 'quiz'}-results.csv"

    def csrf_error_response(status_code: int = 400):
        message = "Invalid or missing security token. Refresh the page and try again."
        if request.endpoint in {"send_invitations", "create_quiz_ai_preview", "detect_face"} or request.is_json or request.headers.get("X-CSRF-Token"):
            return jsonify({"success": False, "message": message}), status_code
        return Response(message, status=status_code, mimetype="text/plain")

    @app.before_request
    def enforce_csrf_protection():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        if request.endpoint == "static":
            return None

        session_token = str(session.get("_csrf_token", "")).strip()
        request_token = (
            request.headers.get("X-CSRF-Token", "")
            or request.form.get("_csrf_token", "")
        ).strip()
        if not session_token or not request_token or not secrets.compare_digest(session_token, request_token):
            return csrf_error_response()
        return None

    @app.before_request
    def synchronize_runtime_state():
        if request.endpoint == "static":
            return None
        sync_quiz_statuses()
        return None

    @app.after_request
    def disable_static_cache(response):
        if app.debug or os.getenv("FLASK_ENV", "development") != "production":
            if request.endpoint == "static" and response.status_code == 200:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                response.headers.pop("ETag", None)
                response.headers.pop("Last-Modified", None)
        return response

    @app.get("/favicon.ico")
    def favicon():
        favicon_svg = """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">
  <rect width=\"64\" height=\"64\" rx=\"14\" fill=\"#8a1538\"/>
  <path d=\"M20 16 L24 8 L40 8 L44 16 Q48 20 48 28 L16 28 Q16 20 20 16\" fill=\"#ffffff\"/>
  <circle cx=\"32\" cy=\"38\" r=\"14\" fill=\"#f7efcf\"/>
  <path d=\"M25 38 Q32 44 39 38\" stroke=\"#8a1538\" stroke-width=\"2.5\" fill=\"none\" stroke-linecap=\"round\"/>
</svg>"""
        return Response(favicon_svg, mimetype="image/svg+xml")

    detection_model = None
    detection_model_error: str | None = None
    detection_model_labels = extract_checkpoint_labels(DETECTION_MODEL_PATH)
    password_reset_serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])
    password_reset_salt = "password-reset"
    password_reset_token_max_age = int(os.environ.get("PASSWORD_RESET_TOKEN_MAX_AGE", "3600"))

    def _should_log_detection_event(attempt_id: str, event_key: str, cooldown_seconds: int = 8) -> bool:
        cache_key = f"{attempt_id}:{event_key}"
        now = datetime.now()
        previous = _detection_event_cache.get(cache_key)
        if previous and (now - previous).total_seconds() < cooldown_seconds:
            return False
        _detection_event_cache[cache_key] = now

        if len(_detection_event_cache) > 400:
            cutoff = now - timedelta(minutes=10)
            stale_keys = [key for key, value in _detection_event_cache.items() if value < cutoff]
            for key in stale_keys:
                _detection_event_cache.pop(key, None)
        return True

    def get_detection_model():
        nonlocal detection_model, detection_model_error
        if detection_model is not None:
            return detection_model
        if detection_model_error:
            raise RuntimeError(detection_model_error)
        if YOLO is None:
            detection_model_error = "Ultralytics YOLO is not installed. Install the required dependencies and restart the server."
            raise RuntimeError(detection_model_error)
        if not DETECTION_MODEL_PATH.exists():
            detection_model_error = f"YOLO model not found at {DETECTION_MODEL_PATH}"
            raise RuntimeError(detection_model_error)
        try:
            detection_model = YOLO(str(DETECTION_MODEL_PATH))
        except Exception as exc:
            labels_hint = f" Embedded labels: {', '.join(detection_model_labels)}." if detection_model_labels else ""
            detection_model_error = f"Unable to load YOLO model from {DETECTION_MODEL_PATH.name}: {exc}.{labels_hint}"
            raise RuntimeError(detection_model_error) from exc
        return detection_model

    def get_detection_runtime_status() -> tuple[bool, str]:
        try:
            get_detection_model()
            labels_hint = f" Labels: {', '.join(detection_model_labels)}." if detection_model_labels else ""
            return True, f"Monitoring model ready: {DETECTION_MODEL_PATH.name}.{labels_hint}"
        except RuntimeError as exc:
            return False, str(exc)

    def _is_google_oauth_configured() -> bool:
        return bool(os.environ.get("GOOGLE_CLIENT_ID", "").strip() and os.environ.get("GOOGLE_CLIENT_SECRET", "").strip())

    def _is_local_host(hostname: str) -> bool:
        host = (hostname or "").strip().lower()
        return host.startswith("localhost") or host.startswith("127.0.0.1")

    def _current_external_url(endpoint: str, **values) -> str:
        """Build external URLs from forwarded headers so ngrok and proxy users get usable links."""
        forwarded_proto = (request.headers.get("X-Forwarded-Proto", "") or "").split(",")[0].strip().lower()
        forwarded_host = (request.headers.get("X-Forwarded-Host", "") or "").split(",")[0].strip()

        scheme = forwarded_proto or request.scheme or "https"
        host = forwarded_host or request.host

        # Normalize default ports to avoid redirect_uri mismatch.
        if scheme == "https" and host.endswith(":443"):
            host = host[:-4]
        elif scheme == "http" and host.endswith(":80"):
            host = host[:-3]

        route_path = url_for(endpoint, _external=False, **values)
        return f"{scheme}://{host}{route_path}"

    def _current_external_callback_uri() -> str:
        return _current_external_url("google_callback")

    def _get_google_oauth_config():
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise RuntimeError("Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")

        configured_redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
        dynamic_redirect_uri = _current_external_callback_uri()
        current_host_is_local = _is_local_host(request.host)

        if current_host_is_local and configured_redirect_uri:
            redirect_uri = configured_redirect_uri
        else:
            redirect_uri = dynamic_redirect_uri

        return client_id, client_secret, redirect_uri

    def _authorize_google(role: str) -> str:
        client_id, _, redirect_uri = _get_google_oauth_config()
        normalized_role = role if role in {"admin", "user"} else "user"
        oauth_state = secrets.token_urlsafe(24)
        session["google_oauth_state"] = {
            "value": oauth_state,
            "role": normalized_role,
        }
        query = urllib_parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "prompt": "select_account",
                "access_type": "offline",
                "state": oauth_state,
            }
        )
        return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"

    def _fetch_google_user_info(code: str) -> dict:
        client_id, client_secret, redirect_uri = _get_google_oauth_config()
        token_request = urllib_request.Request(
            "https://oauth2.googleapis.com/token",
            data=urllib_parse.urlencode(
                {
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(token_request, timeout=20) as response:
                token_data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("Unable to obtain Google token. Please try again.") from exc
        access_token = token_data.get("access_token")
        if not access_token:
            raise RuntimeError("Google did not provide an access token.")
        profile_request = urllib_request.Request(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        try:
            with urllib_request.urlopen(profile_request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("Unable to retrieve Google profile information.") from exc

    class EmailDeliveryConnectivityError(RuntimeError):
        pass

    def _running_on_render() -> bool:
        return any(
            str(os.environ.get(name, "")).strip()
            for name in ("RENDER", "RENDER_SERVICE_ID", "RENDER_EXTERNAL_URL")
        )

    def _send_smtp_email(recipient_email: str, subject: str, body_lines: list[str], *, error_context: str) -> None:
        def env_flag(name: str, default: bool) -> bool:
            raw_value = os.environ.get(name)
            if raw_value is None or not str(raw_value).strip():
                return default
            return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

        def connectivity_error_message(exc: BaseException) -> str | None:
            error_text = str(exc).strip() or exc.__class__.__name__
            lowered_text = error_text.lower()
            error_no = getattr(exc, "errno", None)
            if error_no in {101, 111, 113} or "network is unreachable" in lowered_text or "connection refused" in lowered_text:
                if _running_on_render():
                    return (
                        "SMTP is unreachable from this Render service. On Render free instances, outbound SMTP ports can be blocked. "
                        "Use a paid instance or configure Brevo with EMAIL_DELIVERY_PROVIDER=brevo, BREVO_API_KEY, and EMAIL_FROM."
                    )
                return f"Unable to reach the SMTP server while {error_context}. Verify SMTP_HOST, SMTP_PORT, and outbound network access."
            return None

        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", "").strip() or os.environ.get("EMAIL_FROM", "").strip() or smtp_user or "noreply@example.com"
        try:
            smtp_port = int((os.environ.get("SMTP_PORT", "587") or "587").strip())
        except ValueError as exc:
            raise RuntimeError("SMTP_PORT must be a valid number.") from exc
        try:
            smtp_timeout = float((os.environ.get("SMTP_TIMEOUT", "20") or "20").strip())
        except ValueError as exc:
            raise RuntimeError("SMTP_TIMEOUT must be a valid number.") from exc

        smtp_use_ssl = env_flag("SMTP_USE_SSL", smtp_port == 465)
        smtp_use_tls = env_flag("SMTP_USE_TLS", smtp_port in {587, 2525})
        smtp_require_auth = env_flag("SMTP_REQUIRE_AUTH", bool(smtp_user or smtp_password))

        if smtp_use_ssl:
            smtp_use_tls = False

        if not smtp_host:
            raise RuntimeError("Email sending is not configured. Set SMTP_HOST first.")
        if smtp_require_auth and (not smtp_user or not smtp_password):
            raise RuntimeError("SMTP authentication is enabled, but SMTP_USER or SMTP_PASSWORD is missing.")

        message = EmailMessage()

        effective_sender = smtp_from
        if "gmail.com" in smtp_host.lower() or "googlemail.com" in smtp_host.lower():
            effective_sender = smtp_user
            if smtp_from.lower() != smtp_user.lower():
                message["Reply-To"] = smtp_from

        message["From"] = effective_sender
        message["To"] = recipient_email
        message["Subject"] = subject
        message.set_content("\n".join(body_lines))

        context = ssl.create_default_context()
        smtp_client = None
        try:
            if smtp_use_ssl:
                smtp_client = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=smtp_timeout, context=context)
            else:
                smtp_client = smtplib.SMTP(smtp_host, smtp_port, timeout=smtp_timeout)
            if smtp_use_tls:
                smtp_client.ehlo()
                try:
                    smtp_client.starttls(context=context)
                except smtplib.SMTPNotSupportedError as exc:
                    raise RuntimeError(
                        "The SMTP server does not support STARTTLS. Set SMTP_USE_TLS=0 or use SMTP_PORT=465 with SMTP_USE_SSL=1."
                    ) from exc
                smtp_client.ehlo()
            if smtp_require_auth:
                smtp_client.login(smtp_user, smtp_password)
            smtp_client.send_message(message, from_addr=effective_sender)
        except OSError as exc:
            user_facing_message = connectivity_error_message(exc)
            if user_facing_message:
                raise EmailDeliveryConnectivityError(user_facing_message) from exc
            raise RuntimeError(f"Unable to connect to the SMTP server while {error_context}: {exc}") from exc
        except smtplib.SMTPAuthenticationError as exc:
            raise RuntimeError(
                "SMTP authentication failed. If you are using Gmail, set SMTP_USER to the Gmail address "
                "and SMTP_PASSWORD to a Google App Password."
            ) from exc
        except smtplib.SMTPSenderRefused as exc:
            raise RuntimeError(
                "SMTP rejected the sender address. Set SMTP_FROM to the same mailbox as SMTP_USER, "
                "or use a sender allowed by your mail provider."
            ) from exc
        except smtplib.SMTPException as exc:
            raise RuntimeError(f"SMTP error while {error_context}: {exc}") from exc
        finally:
            try:
                if smtp_client:
                    smtp_client.quit()
            except Exception:
                pass

    def _send_resend_email(recipient_email: str, subject: str, body_lines: list[str], *, error_context: str) -> None:
        resend_api_key = os.environ.get("RESEND_API_KEY", "").strip()
        email_from = os.environ.get("EMAIL_FROM", "").strip() or os.environ.get("SMTP_FROM", "").strip()
        resend_user_agent = os.environ.get("EMAIL_USER_AGENT", "").strip() or "SEMCDS/1.0"

        if not resend_api_key:
            raise RuntimeError("Resend email delivery is not configured. Set RESEND_API_KEY first.")
        if not email_from:
            raise RuntimeError("Set EMAIL_FROM for Resend email delivery.")

        payload = {
            "from": email_from,
            "to": [recipient_email],
            "subject": subject,
            "text": "\n".join(body_lines),
        }
        resend_request = urllib_request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json",
                "User-Agent": resend_user_agent,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(resend_request, timeout=30) as response:
                response.read()
        except urllib_error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            try:
                parsed_body = json.loads(raw_body) if raw_body else {}
            except ValueError:
                parsed_body = {}
            detail = parsed_body.get("message") or raw_body or str(exc)
            raise RuntimeError(f"Resend rejected the email request while {error_context}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Unable to reach the Resend API while {error_context}: {exc.reason}") from exc

    def _send_brevo_email(recipient_email: str, subject: str, body_lines: list[str], *, error_context: str) -> None:
        brevo_api_key = os.environ.get("BREVO_API_KEY", "").strip()
        email_from = os.environ.get("EMAIL_FROM", "").strip() or os.environ.get("SMTP_FROM", "").strip()
        email_from_name = os.environ.get("EMAIL_FROM_NAME", "").strip() or "SEMCDS"
        email_user_agent = os.environ.get("EMAIL_USER_AGENT", "").strip() or "SEMCDS/1.0"

        if not brevo_api_key:
            raise RuntimeError("Brevo email delivery is not configured. Set BREVO_API_KEY first.")
        if not email_from:
            raise RuntimeError("Set EMAIL_FROM for Brevo email delivery.")

        payload = {
            "sender": {
                "email": email_from,
                "name": email_from_name,
            },
            "to": [{"email": recipient_email}],
            "subject": subject,
            "textContent": "\n".join(body_lines),
        }
        brevo_request = urllib_request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                "api-key": brevo_api_key,
                "content-type": "application/json",
                "User-Agent": email_user_agent,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(brevo_request, timeout=30) as response:
                response.read()
        except urllib_error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            try:
                parsed_body = json.loads(raw_body) if raw_body else {}
            except ValueError:
                parsed_body = {}
            detail_parts = [parsed_body.get("message"), parsed_body.get("code")]
            detail = " - ".join(part for part in detail_parts if part) or raw_body or str(exc)
            raise RuntimeError(f"Brevo rejected the email request while {error_context}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Unable to reach the Brevo API while {error_context}: {exc.reason}") from exc

    def _send_email(recipient_email: str, subject: str, body_lines: list[str], *, error_context: str) -> None:
        provider = os.environ.get("EMAIL_DELIVERY_PROVIDER", "auto").strip().lower() or "auto"
        brevo_api_key = os.environ.get("BREVO_API_KEY", "").strip()
        resend_api_key = os.environ.get("RESEND_API_KEY", "").strip()
        smtp_host = os.environ.get("SMTP_HOST", "").strip()

        if provider == "brevo":
            _send_brevo_email(recipient_email, subject, body_lines, error_context=error_context)
            return

        if provider == "resend":
            _send_resend_email(recipient_email, subject, body_lines, error_context=error_context)
            return

        if provider == "smtp":
            _send_smtp_email(recipient_email, subject, body_lines, error_context=error_context)
            return

        if provider not in {"auto", "smtp", "resend", "brevo"}:
            raise RuntimeError("EMAIL_DELIVERY_PROVIDER must be auto, smtp, brevo, or resend.")

        if smtp_host:
            try:
                _send_smtp_email(recipient_email, subject, body_lines, error_context=error_context)
                return
            except EmailDeliveryConnectivityError:
                if brevo_api_key:
                    _send_brevo_email(recipient_email, subject, body_lines, error_context=error_context)
                    return
                if resend_api_key:
                    _send_resend_email(recipient_email, subject, body_lines, error_context=error_context)
                    return
                raise

        if brevo_api_key:
            _send_brevo_email(recipient_email, subject, body_lines, error_context=error_context)
            return

        if resend_api_key:
            _send_resend_email(recipient_email, subject, body_lines, error_context=error_context)
            return

        raise RuntimeError(
            "Email sending is not configured. Set SMTP_HOST for SMTP, or set BREVO_API_KEY with EMAIL_FROM for Brevo, or set RESEND_API_KEY with EMAIL_FROM for Resend."
        )

    def _send_invitation_email(recipient_email: str, temporary_password: str | None = None) -> None:
        login_url = _current_external_url("login")
        body_lines = [
            "Hello,",
            "",
            "You've been invited to join SEMCDS as a Student.",
            "",
        ]
        if temporary_password:
            body_lines.extend(
                [
                    "A new account has been created for you.",
                    "",
                    f"Email: {recipient_email}",
                    f"Temporary password: {temporary_password}",
                    "",
                ]
            )
        body_lines.extend(
            [
                f"Sign in here: {login_url}",
                "",
                "If you did not expect this invitation, please ignore this message.",
            ]
        )
        _send_email(
            recipient_email,
            "You're invited to SEMCDS",
            body_lines,
            error_context="sending invitation email",
        )

    def _send_password_reset_email(recipient_email: str, reset_url: str) -> None:
        _send_email(
            recipient_email,
            "SEMCDS password reset",
            [
                "Hello,",
                "",
                "We received a request to reset your SEMCDS password.",
                f"Reset your password using this secure link: {reset_url}",
                "",
                "This link expires in 60 minutes and becomes invalid after you change your password.",
                "If you did not request a reset, you can ignore this email.",
            ],
            error_context="sending password reset email",
        )

    def _password_reset_token_signature(user: dict) -> str:
        payload = "|".join(
            [
                str(user.get("id", "")).strip(),
                str(user.get("email", "")).strip().lower(),
                str(user.get("password_hash", "")).strip(),
                app.config["SECRET_KEY"],
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_password_reset_token(user: dict) -> str:
        return password_reset_serializer.dumps(
            {
                "user_id": user.get("id", ""),
                "email": user.get("email", ""),
                "signature": _password_reset_token_signature(user),
            },
            salt=password_reset_salt,
        )

    def _read_password_reset_token(token: str) -> dict | None:
        try:
            payload = password_reset_serializer.loads(
                token,
                salt=password_reset_salt,
                max_age=password_reset_token_max_age,
            )
        except (BadSignature, SignatureExpired):
            return None
        if not isinstance(payload, dict):
            return None
        user_id = str(payload.get("user_id", "")).strip()
        email = str(payload.get("email", "")).strip().lower()
        signature = str(payload.get("signature", "")).strip().lower()
        if not user_id or not email or not signature:
            return None
        return {"user_id": user_id, "email": email, "signature": signature}

    def decode_image_from_data_url(data_url: str) -> np.ndarray | None:
        if not data_url.startswith("data:image"):
            return None
        try:
            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            array = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def normalize_schedule_input(raw_value: str) -> str:
        return raw_value.replace("T", " ").strip()

    def compute_time_limit_minutes(scheduled_start: str, scheduled_end: str, fallback_minutes: int = 0) -> int:
        start = parse_schedule(scheduled_start)
        end = parse_schedule(scheduled_end)
        if start and end and end > start:
            minutes = int((end - start).total_seconds() // 60)
            return max(1, minutes)
        return max(0, fallback_minutes)

    def blank_quiz() -> dict:
        return {
            "id": "",
            "title": "",
            "description": "",
            "subject": "",
            "assigned_section": "",
            "time_limit_minutes": 0,
            "status": "draft",
            "quiz_code": "",
            "monitoring_enabled": False,
            "scheduled_start": "",
            "scheduled_end": "",
            "questions": [],
            "total_points": 0,
        }

    def available_student_sections(selected_section: str = "") -> list[str]:
        sections = list(QUIZ_SECTION_OPTIONS)
        seen = {section.casefold() for section in sections}
        selected_clean = " ".join(str(selected_section or "").split())
        if selected_clean and selected_clean.casefold() not in seen:
            sections.append(selected_clean)
        return sections

    def student_identity_key(student_name: str, student_email: str) -> str:
        normalized_email = student_email.strip().lower()
        return normalized_email or f"unknown:{student_name.strip().lower()}"

    def is_live_quiz(quiz: dict | None, now: datetime | None = None) -> bool:
        if not quiz:
            return False
        return quiz.get("status") == "published" and schedule_status(quiz, now) == "open"

    def connected_student_sessions_for_quiz(quiz: dict, now: datetime | None = None) -> list[dict]:
        current_time = now or datetime.now()
        active_attempts = {
            str(attempt.get("id", "")): attempt
            for attempt in quiz_attempts(quiz["id"])
            if attempt.get("status") == "in_progress" and not attempt_has_expired(quiz, attempt, current_time)
        }

        if not socketio:
            return [{"attempt": attempt, "participant": None} for attempt in active_attempts.values()]

        room_key = f"monitor:{quiz['id']}"
        participants = _monitor_rooms.get(room_key, {})
        sessions: list[dict] = []
        seen_attempt_ids: set[str] = set()
        for participant in participants.values():
            if participant.get("role") != "user":
                continue
            attempt_id = str(participant.get("attempt_id", "")).strip()
            if not attempt_id or attempt_id in seen_attempt_ids:
                continue
            attempt = active_attempts.get(attempt_id)
            if not attempt:
                continue
            seen_attempt_ids.add(attempt_id)
            sessions.append({"attempt": attempt, "participant": participant})
        return sessions

    def build_in_progress_student_rows(
        quiz_source: dict | list[dict] | None,
        *,
        severity: str = "all",
        reviewed: str = "all",
        search: str = "",
        selected_student_email: str = "",
        now: datetime | None = None,
    ) -> list[dict]:
        if not quiz_source:
            return []

        current_time = now or datetime.now()
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        quiz_list = [quiz_source] if isinstance(quiz_source, dict) else list(quiz_source)
        student_map: dict[str, dict] = {}

        for quiz in quiz_list:
            if not quiz:
                continue
            all_flags = quiz_flags(quiz["id"])
            flags_by_attempt: dict[str, list[dict]] = {}
            for flag in all_flags:
                flags_by_attempt.setdefault(str(flag.get("attempt_id", "")), []).append(flag)

            for session in connected_student_sessions_for_quiz(quiz, current_time):
                attempt = session["attempt"]
                student_name = str(attempt.get("student_name", "Unknown Student")).strip() or "Unknown Student"
                student_email = str(attempt.get("student_email", "")).strip().lower()
                student_key = student_identity_key(student_name, student_email)

                if selected_student_email and student_key != selected_student_email:
                    continue
                if search and search not in student_name.lower() and search not in student_email.lower():
                    continue

                attempt_flags = flags_by_attempt.get(str(attempt.get("id", "")), [])
                matching_flags = list(attempt_flags)
                if severity != "all":
                    matching_flags = [flag for flag in matching_flags if flag.get("flag_level") == severity]
                if reviewed != "all":
                    expected_reviewed = reviewed == "reviewed"
                    matching_flags = [flag for flag in matching_flags if bool(flag.get("reviewed")) == expected_reviewed]

                if (severity != "all" or reviewed != "all") and not matching_flags:
                    continue

                matching_flags.sort(key=lambda flag: sort_timestamp(flag.get("timestamp")), reverse=True)
                latest_flag = matching_flags[0] if matching_flags else (attempt_flags[0] if attempt_flags else None)
                entry = student_map.setdefault(
                    student_key,
                    {
                        "student_key": student_key,
                        "student_name": student_name,
                        "student_email": student_email,
                        "attempt_id": str(attempt.get("id", "")),
                        "event_count": 0,
                        "quiz_ids": set(),
                        "highest_level": "low",
                        "pending_count": 0,
                        "reviewed_count": 0,
                        "event_counter": Counter(),
                        "latest_event": "Joined quiz",
                        "latest_quiz": quiz.get("title", ""),
                        "last_activity": "Currently Active",
                        "status": "in_progress",
                        "has_matching_activity": False,
                    },
                )

                entry["quiz_ids"].add(quiz["id"])
                entry["event_count"] += len(matching_flags)
                entry["pending_count"] += sum(1 for flag in matching_flags if not flag.get("reviewed"))
                entry["reviewed_count"] += sum(1 for flag in matching_flags if flag.get("reviewed"))
                entry["event_counter"].update(activity_event_label(flag.get("event_type", "")) for flag in matching_flags)
                entry["has_matching_activity"] = entry["has_matching_activity"] or bool(matching_flags)
                if latest_flag:
                    entry["latest_event"] = activity_event_label(latest_flag.get("event_type", ""))
                    entry["latest_quiz"] = quiz.get("title", "")
                    entry["last_activity"] = latest_flag.get("timestamp", "") or entry["last_activity"]

                for flag in matching_flags:
                    current_rank = severity_rank.get(flag.get("flag_level", "low"), 1)
                    saved_rank = severity_rank.get(entry["highest_level"], 1)
                    if current_rank >= saved_rank:
                        entry["highest_level"] = str(flag.get("flag_level", "low"))

        rows = [
            {
                "student_key": item["student_key"],
                "student_name": item["student_name"],
                "student_email": item["student_email"],
                "attempt_id": item["attempt_id"],
                "event_count": item["event_count"],
                "quiz_count": len(item["quiz_ids"]),
                "highest_level": item["highest_level"],
                "pending_count": item["pending_count"],
                "reviewed_count": item["reviewed_count"],
                "event_tallies": item["event_counter"].most_common(3),
                "latest_event": item["latest_event"],
                "latest_quiz": item["latest_quiz"],
                "last_activity": item["last_activity"],
                "status": item["status"],
                "has_matching_activity": item["has_matching_activity"],
            }
            for item in student_map.values()
        ]
        rows.sort(
            key=lambda item: (
                -severity_rank.get(item["highest_level"], 1),
                -item["event_count"],
                item["student_name"].lower(),
            )
        )
        return rows

    def chunk_source_text(raw_text: str) -> list[str]:
        return [
            item.strip()
            for item in raw_text.replace("\r", " ").replace("\n", " ").split(". ")
            if item.strip() and len(item.strip()) > 25
        ]

    def build_fallback_lesson_text(file_name: str) -> str:
        lesson_name = (
            (file_name or "the uploaded lesson")
            .replace(".pdf", "")
            .replace(".txt", "")
            .replace("_", " ")
            .replace("-", " ")
            .strip()
        )
        return " ".join(
            [
                f"{lesson_name} covers the main concepts discussed in the uploaded material.",
                f"Important definitions, examples, and review points are included in {lesson_name}.",
                f"Students are expected to understand the key ideas and supporting details from {lesson_name}.",
            ]
        )

    def split_sentences(raw_text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", raw_text.replace("\r", " ").replace("\n", " ")).strip()
        sentences = re.split(r"(?<=[\.\!\?])\s+", normalized)
        cleaned: list[str] = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip(" -•\t")
            if len(sentence) < 35:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(sentence)
        return cleaned

    def shorten_text(raw_text: str, max_words: int = 14, max_chars: int = 90) -> str:
        text = re.sub(r"\s+", " ", raw_text).strip(" .,:;")
        words = text.split()
        shortened = " ".join(words[:max_words])
        if len(shortened) > max_chars:
            shortened = shortened[: max_chars - 1].rsplit(" ", 1)[0]
        return shortened.strip(" ,;:.") or text[:max_chars].strip(" ,;:.")

    def derive_keyword_phrase(sentence: str) -> str:
        keyword_candidates = re.findall(r"\b[A-Za-z][A-Za-z\-]{3,}\b", sentence)
        filtered = []
        stopwords = {
            "which",
            "these",
            "those",
            "their",
            "there",
            "about",
            "because",
            "during",
            "after",
            "before",
            "using",
            "includes",
            "important",
            "students",
            "expected",
            "discussed",
            "material",
            "lesson",
            "topic",
        }
        for word in keyword_candidates:
            lower = word.lower()
            if lower in stopwords:
                continue
            filtered.append(word)
        if not filtered:
            return shorten_text(sentence, max_words=6, max_chars=48)
        return " ".join(filtered[:4])

    def extract_upload_text(uploaded_file) -> tuple[str, str]:
        filename = (uploaded_file.filename or "").strip()
        lower_name = filename.lower()
        if lower_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return text.strip(), "Text file loaded successfully."

        if lower_name.endswith(".pdf"):
            reader = PdfReader(BytesIO(uploaded_file.read()))
            extracted_pages = []
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    extracted_pages.append(page_text)
            clean_text = " ".join(extracted_pages).strip()
            if clean_text:
                return clean_text, "PDF loaded successfully."
            return build_fallback_lesson_text(filename), "The PDF text could not be extracted clearly, so a clean fallback preview was generated from the file name."

        return "", "Please upload a PDF or TXT file."

    def generate_questions_locally(raw_text: str, requested_count: int, requested_type: str) -> list[dict]:
        chunks = split_sentences(raw_text)
        base_chunks = chunks or ["The uploaded file contains lesson content for quiz generation."]
        count = max(1, min(int(requested_count or 5), 30))
        questions: list[dict] = []

        for index in range(count):
            source = base_chunks[index % len(base_chunks)]
            normalized_source = shorten_text(source, max_words=22, max_chars=170)
            question_type = requested_type
            if requested_type == "mixed":
                question_type = "multiple_choice" if index % 2 == 0 else "true_false"

            if question_type == "true_false":
                questions.append(
                    {
                        "question_text": f"True or False: {normalized_source}",
                        "question_type": "true_false",
                        "points": 1,
                        "options": ["True", "False"],
                        "correct_answer": "True",
                    }
                )
            else:
                correct_option = derive_keyword_phrase(source)
                distractor_sources = [
                    derive_keyword_phrase(base_chunks[(index + offset) % len(base_chunks)])
                    for offset in range(1, 5)
                ]
                distractors = []
                for item in distractor_sources:
                    cleaned = shorten_text(item, max_words=8, max_chars=56)
                    if cleaned and cleaned.lower() != correct_option.lower() and cleaned not in distractors:
                        distractors.append(cleaned)
                while len(distractors) < 3:
                    distractors.append(f"Related concept {len(distractors) + 1}")
                questions.append(
                    {
                        "question_text": f"What concept is being described in this statement: {normalized_source}?",
                        "question_type": "multiple_choice",
                        "points": 1,
                        "options": [
                            correct_option,
                            distractors[0],
                            distractors[1],
                            distractors[2],
                        ],
                        "correct_answer": correct_option,
                    }
                )

        return questions

    def call_openai_question_generator(raw_text: str, requested_count: int, requested_type: str) -> list[dict]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        source_excerpt = raw_text[:12000]
        prompt = (
            "Generate quiz questions from the uploaded lesson content. "
            "Return JSON only. Keep wording natural and concise. "
            f"Question type preference: {requested_type}. "
            f"Number of questions: {max(1, min(int(requested_count or 5), 30))}. "
            "Use only information supported by the source text. "
            "For multiple choice, make all options short and plausible."
            "\n\nSource text:\n"
            f"{source_excerpt}"
        )

        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You generate quiz questions for an exam platform. "
                        "Always output valid JSON that matches the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "quiz_generation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "question_text": {"type": "string"},
                                        "question_type": {"type": "string", "enum": ["multiple_choice", "true_false"]},
                                        "points": {"type": "integer"},
                                        "options": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "correct_answer": {"type": "string"},
                                    },
                                    "required": ["question_text", "question_type", "points", "options", "correct_answer"],
                                },
                            }
                        },
                        "required": ["questions"],
                    },
                }
            },
        }

        req = urllib_request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        output_text = body.get("output_text", "")
        if not output_text:
            raise RuntimeError("The AI service returned an empty response.")

        parsed = json.loads(output_text)
        questions = parsed.get("questions", [])
        cleaned_questions = []
        for item in questions:
            question_type = item.get("question_type", "multiple_choice")
            options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
            if question_type == "true_false":
                options = ["True", "False"]
            elif len(options) < 2:
                continue
            cleaned_questions.append(
                {
                    "question_text": str(item.get("question_text", "")).strip(),
                    "question_type": question_type,
                    "points": int(item.get("points", 1) or 1),
                    "options": options,
                    "correct_answer": str(item.get("correct_answer", options[0] if options else "")).strip(),
                }
            )
        if not cleaned_questions:
            raise RuntimeError("The AI service did not return usable questions.")
        return cleaned_questions

    def call_gemini_question_generator(raw_text: str, requested_count: int, requested_type: str) -> list[dict]:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
        source_excerpt = raw_text[:12000]
        prompt = (
            "Generate quiz questions from the uploaded lesson content. "
            "Keep the wording natural, concise, and classroom-appropriate. "
            f"Question type preference: {requested_type}. "
            f"Number of questions: {max(1, min(int(requested_count or 5), 30))}. "
            "Use only information supported by the source text. "
            "For multiple choice, keep options short and plausible."
            "\n\nSource text:\n"
            f"{source_excerpt}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "question_text": {"type": "string"},
                                    "question_type": {"type": "string", "enum": ["multiple_choice", "true_false"]},
                                    "points": {"type": "integer"},
                                    "options": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "correct_answer": {"type": "string"},
                                },
                                "required": ["question_text", "question_type", "points", "options", "correct_answer"],
                            },
                        }
                    },
                    "required": ["questions"],
                },
            },
        }

        req = urllib_request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        candidates = body.get("candidates", [])
        if not candidates:
            raise RuntimeError("The Gemini service returned no candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text_part = "".join(part.get("text", "") for part in parts if part.get("text"))
        if not text_part:
            raise RuntimeError("The Gemini service returned an empty response.")

        parsed = json.loads(text_part)
        questions = parsed.get("questions", [])
        cleaned_questions = []
        for item in questions:
            question_type = item.get("question_type", "multiple_choice")
            options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
            if question_type == "true_false":
                options = ["True", "False"]
            elif len(options) < 2:
                continue
            cleaned_questions.append(
                {
                    "question_text": str(item.get("question_text", "")).strip(),
                    "question_type": question_type,
                    "points": int(item.get("points", 1) or 1),
                    "options": options,
                    "correct_answer": str(item.get("correct_answer", options[0] if options else "")).strip(),
                }
            )
        if not cleaned_questions:
            raise RuntimeError("The Gemini service did not return usable questions.")
        return cleaned_questions

    def generate_questions_from_text(raw_text: str, requested_count: int, requested_type: str) -> tuple[list[dict], str]:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if gemini_api_key:
            try:
                return call_gemini_question_generator(raw_text, requested_count, requested_type), "Real Gemini AI question generation was used for this preview."
            except (RuntimeError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError, TimeoutError, ValueError) as exc:
                if api_key:
                    try:
                        return call_openai_question_generator(raw_text, requested_count, requested_type), f"Gemini generation could not be completed, so OpenAI was used instead. ({exc})"
                    except (RuntimeError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError, TimeoutError, ValueError) as openai_exc:
                        fallback_questions = generate_questions_locally(raw_text, requested_count, requested_type)
                        return fallback_questions, f"Gemini and OpenAI generation could not be completed, so the improved local generator was used instead. ({openai_exc})"
                fallback_questions = generate_questions_locally(raw_text, requested_count, requested_type)
                return fallback_questions, f"Gemini generation could not be completed, so the improved local generator was used instead. ({exc})"

        if api_key:
            try:
                return call_openai_question_generator(raw_text, requested_count, requested_type), "Real AI question generation was used for this preview."
            except (RuntimeError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError, TimeoutError, ValueError) as exc:
                fallback_questions = generate_questions_locally(raw_text, requested_count, requested_type)
                return fallback_questions, f"AI generation could not be completed, so the improved local generator was used instead. ({exc})"

        return generate_questions_locally(raw_text, requested_count, requested_type), "Improved local question generation was used. Add GEMINI_API_KEY or OPENAI_API_KEY to enable real AI generation."

    @app.context_processor
    def inject_globals():
        current_user = current_session_user()
        role = current_user.get("role") if current_user else None
        return {
            "current_role": role,
            "current_user": current_user,
            "current_endpoint": request.endpoint,
        }

    def role_required(*allowed_roles):
        def decorator(view):
            @wraps(view)
            def wrapped(*args, **kwargs):
                current_user = current_session_user()
                role = current_user.get("role") if current_user else None
                if not role:
                    return redirect_to("login")
                if role not in allowed_roles:
                    return redirect_to("home")
                return view(*args, **kwargs)

            return wrapped

        return decorator

    @app.route("/")
    def index():
        if current_session_user():
            return redirect_to("home")
        return render_template(
            "login.html",
            selected_role="admin",
            error="",
            forgot_message="",
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        selected_role = request.values.get("role", request.args.get("role", "admin")).strip()
        if selected_role not in {"admin", "user"}:
            selected_role = "admin"

        error = request.args.get("error", "").strip()
        forgot_message = request.args.get("message", "").strip()

        if request.method == "POST":
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            selected_role = request.form.get("role", selected_role).strip()
            remember_me = request.form.get("remember_me") == "on"
            user = get_user_by_email(email)

            if not verify_password(user, password):
                error = "Invalid email or password."
            elif user["role"] != selected_role:
                error = "This account does not match the selected portal."
            else:
                session.clear()
                session["role"] = user["role"]
                session["user_id"] = user["id"]
                session["avatar_url"] = _normalize_avatar_url(user.get("avatar_url"))
                session.permanent = remember_me
                issue_csrf_token(force=True)
                return redirect(url_for("home", login_success="1"))

        return render_template(
            "login.html",
            selected_role=selected_role,
            error=error,
            forgot_message=forgot_message,
        )

    @app.route("/google-login")
    def google_login():
        role = request.args.get("role", "user").strip()
        if role not in {"admin", "user"}:
            role = "user"
        if not _is_google_oauth_configured():
            return redirect_to("login", role=role, error="Google OAuth is not configured.")
        try:
            return redirect(_authorize_google(role))
        except RuntimeError as exc:
            return redirect_to("login", role=role, error=str(exc))

    @app.route("/google-callback")
    def google_callback():
        code = request.args.get("code", "").strip()
        callback_state = request.args.get("state", "").strip()
        oauth_state = session.pop("google_oauth_state", None)
        role = str((oauth_state or {}).get("role", "user")).strip()
        if role not in {"admin", "user"}:
            role = "user"
        if not oauth_state or not callback_state or not secrets.compare_digest(str(oauth_state.get("value", "")), callback_state):
            return redirect_to("login", role=role, error="Google login session expired. Please try again.")
        if not code:
            return redirect_to("login", role=role, error="Google login failed.")
        try:
            profile = _fetch_google_user_info(code)
        except Exception as exc:
            return redirect_to("login", role=role, error=str(exc))

        email = (profile.get("email") or "").strip().lower()
        full_name = (profile.get("name") or profile.get("given_name") or email).strip()
        avatar_url = _normalize_avatar_url(profile.get("picture"))
        if not email:
            return redirect_to("login", role=role, error="Google did not return a valid email address.")

        user = get_user_by_email(email)
        if user and user["role"] != role:
            return redirect_to("login", role=role, error="Your Google account does not match the selected portal.")

        if not user:
            if role == "admin":
                return redirect_to(
                    "login",
                    role=role,
                    error="This instructor Google account is not authorized. Only registered instructor accounts can sign in.",
                )
            return redirect_to(
                "login",
                role=role,
                error="This student Google account is not invited yet. Ask your instructor to send an invitation first.",
            )

        stored_avatar_url = _normalize_avatar_url(user.get("avatar_url"))
        if avatar_url and avatar_url != stored_avatar_url:
            set_user_avatar_url(user["id"], avatar_url)
            stored_avatar_url = avatar_url

        session.clear()
        session["role"] = role
        session["user_id"] = user["id"]
        session["avatar_url"] = avatar_url or stored_avatar_url
        issue_csrf_token(force=True)
        return redirect(url_for("home", login_success="1"))

    @app.route("/send-invitations", methods=["POST"])
    @role_required("admin")
    def send_invitations():
        raw_emails = request.form.get("emails", "").strip()
        if not raw_emails:
            return jsonify({"success": False, "message": "Provide at least one email address."}), 400

        candidate_emails = {
            email.strip().lower()
            for email in re.split(r"[,;\n]+", raw_emails)
            if email.strip()
        }
        if not candidate_emails:
            return jsonify({"success": False, "message": "Provide at least one valid email address."}), 400

        sent = []
        errors = []
        for email in sorted(candidate_emails):
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                errors.append({"email": email, "message": "Invalid email format."})
                continue

            existing_user = get_user_by_email(email)
            created = False
            temporary_password = None
            if not existing_user:
                created = True
                temporary_password = os.urandom(10).hex()
                create_user(email, email.split("@")[0].replace(".", " ").title(), "user", temporary_password)

            try:
                _send_invitation_email(email, temporary_password)
                sent.append({"email": email, "created": created})
            except Exception as exc:
                if created:
                    delete_user_by_email(email)
                errors.append({"email": email, "message": str(exc)})

        if not sent:
            return jsonify({"success": False, "message": "Unable to send any invitations.", "errors": errors}), 500

        return jsonify({"success": True, "message": f"Invitations sent to {len(sent)} recipient(s).", "sent": sent, "errors": errors})

    @app.route("/forgot-password", methods=["GET", "POST"])
    def forgot_password():
        message = ""
        error = ""
        email_value = request.values.get("email", "").strip().lower()

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            email_value = email
            if not email:
                error = "Enter your email address."
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                error = "Enter a valid email address."
            else:
                user = get_user_by_email(email)
                if user:
                    try:
                        reset_token = _build_password_reset_token(user)
                        reset_url = _current_external_url("reset_password", token=reset_token)
                        _send_password_reset_email(user["email"], reset_url)
                    except RuntimeError as exc:
                        error = str(exc)
                    except Exception as exc:
                        error = f"Unable to send reset email right now. ({exc})"

                if not error:
                    message = "If the email exists in the system, a password reset link has been sent."

        return render_template(
            "forgot_password.html",
            message=message,
            error=error,
            email_value=email_value,
        )

    @app.route("/reset-password/<token>", methods=["GET", "POST"])
    def reset_password(token: str):
        invalid_token_message = "This password reset link is invalid, expired, or has already been used."
        token_payload = _read_password_reset_token(token)
        if not token_payload:
            return render_template(
                "reset_password.html",
                token_valid=False,
                error=invalid_token_message,
                success_message="",
                token=token,
            )

        user = get_user_by_id(token_payload["user_id"])
        current_signature = _password_reset_token_signature(user or {})
        if (
            not user
            or user.get("email", "").strip().lower() != token_payload["email"]
            or not secrets.compare_digest(token_payload["signature"], current_signature)
        ):
            return render_template(
                "reset_password.html",
                token_valid=False,
                error=invalid_token_message,
                success_message="",
                token=token,
            )

        error = ""
        success_message = ""
        if request.method == "POST":
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if len(password) < 8:
                error = "Password must be at least 8 characters long."
            elif password != confirm_password:
                error = "Passwords do not match."
            elif not set_user_password(user["id"], password):
                error = "Unable to reset password right now. Please try again."
            else:
                return redirect(
                    url_for(
                        "login",
                        message="Password reset successful. You can now sign in with your new password.",
                        role=user.get("role", "admin"),
                    )
                )

        return render_template(
            "reset_password.html",
            token_valid=True,
            error=error,
            success_message=success_message,
            token=token,
        )

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login", logout_success="1"))

    @app.route("/home")
    def home():
        current_user = current_session_user()
        role = current_user.get("role") if current_user else None
        login_success = request.args.get("login_success") == "1"
        if role == "admin":
            return redirect_to("dashboard", login_success="1") if login_success else redirect_to("dashboard")
        if role == "user":
            return redirect_to("student_dashboard", login_success="1") if login_success else redirect_to("student_dashboard")
        return redirect_to("login")

    @app.route("/Dashboard")
    @role_required("admin")
    def dashboard():
        current_time = datetime.now()
        message = request.args.get("message", "").strip()
        selected_quiz = request.args.get("quizId", "").strip()
        analyzed = request.args.get("analyzed") == "1"
        all_quizzes = get_quizzes()
        analyzable_quizzes = [quiz for quiz in all_quizzes if quiz["monitoring_enabled"]]
        valid_quiz_ids = {quiz["id"] for quiz in analyzable_quizzes}
        if selected_quiz not in valid_quiz_ids:
            selected_quiz = ""
            analyzed = False
        month_param = request.args.get("month", current_time.strftime("%Y-%m"))
        selected_day_value = parse_dashboard_day_key(request.args.get("day", ""))
        try:
            current_month = datetime.strptime(month_param, "%Y-%m")
        except ValueError:
            current_month = (selected_day_value or current_time).replace(day=1)
        if selected_day_value and (selected_day_value.year != current_month.year or selected_day_value.month != current_month.month):
            current_month = selected_day_value.replace(day=1)
        calendar_rows = build_dashboard_calendar(current_month.year, current_month.month)
        selected_day = selected_day_value.strftime("%Y-%m-%d") if selected_day_value else ""
        selected_day_quizzes = scheduled_quizzes_for_day(selected_day_value) if selected_day_value else []
        prev_month = current_month.month - 1 or 12
        prev_year = current_month.year - 1 if current_month.month == 1 else current_month.year
        next_month = 1 if current_month.month == 12 else current_month.month + 1
        next_year = current_month.year + 1 if current_month.month == 12 else current_month.year
        recent_quizzes = all_quizzes[:4]
        recent_attempts = [attempt for quiz in recent_quizzes for attempt in quiz_attempts(quiz["id"])]
        return render_template(
            "dashboard.html",
            message=message,
            stats=dashboard_stats(),
            recent_quizzes=recent_quizzes,
            summary=cheating_summary(selected_quiz) if selected_quiz and analyzed else None,
            quizzes=analyzable_quizzes,
            attempts=recent_attempts,
            selected_quiz=selected_quiz,
            analyzed=analyzed,
            calendar_rows=calendar_rows,
            calendar_month=current_month.strftime("%B %Y"),
            calendar_month_key=current_month.strftime("%Y-%m"),
            prev_month=f"{prev_year:04d}-{prev_month:02d}",
            next_month=f"{next_year:04d}-{next_month:02d}",
            selected_day=selected_day,
            selected_day_label=format_dashboard_day_label(selected_day_value),
            selected_day_checked_at=format_current_timestamp(current_time) if selected_day_value else "",
            selected_day_quizzes=selected_day_quizzes,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
        )

    @app.route("/api/dashboard/schedule")
    @role_required("admin")
    def dashboard_schedule_api():
        selected_day_value = parse_dashboard_day_key(request.args.get("day", ""))
        if not selected_day_value:
            return jsonify({"message": "Choose a valid schedule date."}), 400

        current_time = datetime.now()
        selected_day_quizzes = scheduled_quizzes_for_day(selected_day_value)
        return jsonify(
            {
                "day": selected_day_value.strftime("%Y-%m-%d"),
                "day_label": format_dashboard_day_label(selected_day_value),
                "month": selected_day_value.strftime("%Y-%m"),
                "month_label": selected_day_value.strftime("%B %Y"),
                "checked_at": format_current_timestamp(current_time),
                "quizzes": [
                    {
                        "id": quiz["id"],
                        "title": quiz.get("title", "Untitled Quiz"),
                        "status": schedule_status(quiz, current_time),
                        "scheduled_start_label": format_schedule(quiz.get("scheduled_start")),
                        "scheduled_end_label": format_schedule(quiz.get("scheduled_end")),
                    }
                    for quiz in selected_day_quizzes
                ],
            }
        ), 200

    @app.route("/Dashboard/ResetData", methods=["POST"])
    @role_required("admin")
    def dashboard_reset_data():
        reset_summary = reset_dashboard_data()
        _monitor_rooms.clear()
        _detection_event_cache.clear()

        if any(reset_summary.values()):
            message = (
                "Dashboard data reset. "
                f"Cleared {reset_summary['attempts']} attempt(s), "
                f"{reset_summary['responses']} response(s), and "
                f"{reset_summary['activity_logs']} activity log(s)."
            )
        else:
            message = "Dashboard data is already clear."
        return redirect_to("dashboard", message=message)

    @app.route("/QuizManager")
    @role_required("admin")
    def quiz_manager():
        user = current_session_user()
        if not user or user.get("role") != "admin":
            return redirect_to("login")
        status_filter = request.args.get("status", "all").strip().lower()
        if status_filter not in {"all", "draft", "published", "closed"}:
            status_filter = "all"
        search = request.args.get("q", "").strip()
        search_term = search.casefold()
        message = request.args.get("message", "").strip()
        all_quizzes = [
            quiz
            for quiz in get_quizzes()
            if str(quiz.get("creator_id", "")).strip() == str(user["id"]).strip()
        ]
        quizzes = all_quizzes if status_filter == "all" else [quiz for quiz in all_quizzes if quiz["status"] == status_filter]
        if search_term:
            quizzes = [
                quiz
                for quiz in quizzes
                if search_term in quiz["title"].casefold()
                or search_term in quiz["subject"].casefold()
                or search_term in quiz["quiz_code"].casefold()
            ]
        quiz_attempt_totals = {
            quiz["id"]: sum(
                1 for attempt in quiz_attempts(quiz["id"])
                if attempt.get("status") in {"submitted", "auto_submitted"}
            )
            for quiz in quizzes
        }
        return render_template(
            "quiz_manager.html",
            quizzes=quizzes,
            status_filter=status_filter,
            search=search,
            message=message,
            has_active_filters=bool(search or status_filter != "all"),
            filtered_quiz_count=len(quizzes),
            total_quizzes_count=len(all_quizzes),
            quiz_attempt_totals=quiz_attempt_totals,
        )

    @app.route("/QuizAction", methods=["POST"])
    @role_required("admin")
    def quiz_action():
        user = current_session_user()
        if not user or user.get("role") != "admin":
            return redirect_to("login")

        quiz_id = request.form.get("quiz_id", "").strip()
        action = request.form.get("action", "").strip()
        message = ""
        owned_quizzes = [
            quiz
            for quiz in get_quizzes()
            if str(quiz.get("creator_id", "")).strip() == str(user["id"]).strip()
        ]

        if action == "clear_all_created":
            if not owned_quizzes:
                message = "You do not have any created quizzes to clear."
            else:
                for quiz in owned_quizzes:
                    delete_quiz_by_id(quiz["id"])
                cleared_count = len(owned_quizzes)
                message = f"Cleared {cleared_count} quiz{'zes' if cleared_count != 1 else ''} you created."
            return redirect_to("quiz_manager", message=message)

        quiz = get_quiz(quiz_id) if quiz_id else None
        if not quiz:
            return redirect_to("quiz_manager", message="Action could not be completed.")

        if str(quiz.get("creator_id", "")).strip() != str(user["id"]).strip():
            return redirect_to("quiz_manager", message="You can only manage quizzes you created.")

        if action == "close":
            set_quiz_status(quiz_id, "closed")
            message = "Quiz closed successfully."
        elif action == "reopen":
            set_quiz_status(quiz_id, "draft")
            return redirect_to("create_quiz", quizId=quiz_id)
        elif action == "delete":
            delete_quiz_by_id(quiz_id)
            message = "Quiz deleted successfully."
        else:
            message = "Action could not be completed."

        return redirect_to("quiz_manager", message=message)

    @app.route("/CreateQuiz/AIPreview", methods=["POST"])
    @role_required("admin")
    def create_quiz_ai_preview():
        uploaded_file = request.files.get("file")
        if not uploaded_file or not (uploaded_file.filename or "").strip():
            return jsonify({"ok": False, "message": "Upload a PDF or TXT file first."}), 400

        question_type = request.form.get("question_type", "mixed").strip() or "mixed"
        try:
            question_count = int(request.form.get("question_count", "5") or 5)
        except ValueError:
            question_count = 5

        extracted_text, status_message = extract_upload_text(uploaded_file)
        if not extracted_text.strip():
            return jsonify({"ok": False, "message": status_message}), 400

        questions, generation_message = generate_questions_from_text(extracted_text, question_count, question_type)
        return jsonify(
            {
                "ok": True,
                "message": f"{status_message} {generation_message}".strip(),
                "questions": questions,
            }
        )

    @app.route("/CreateQuiz", methods=["GET", "POST"])
    @role_required("admin")
    def create_quiz():
        if request.method == "POST":
            quiz_id = request.form.get("quiz_id", "").strip()
            action = request.form.get("action", "draft").strip()
            user = current_session_user()
            if not user or user.get("role") != "admin":
                return redirect_to("login")
            title = request.form.get("title", "").strip()
            description = request.form.get("description", "").strip()
            subject = request.form.get("subject", "").strip()
            assigned_section = request.form.get("assigned_section", "").strip()
            allowed_sections = {section.casefold(): section for section in available_student_sections()}
            if assigned_section:
                assigned_section = allowed_sections.get(assigned_section.casefold(), "")
            quiz_code = request.form.get("quiz_code", "").strip().upper() or f"QUIZ-{datetime.now().strftime('%H%M%S')}"
            scheduled_start = normalize_schedule_input(request.form.get("scheduled_start", ""))
            scheduled_end = normalize_schedule_input(request.form.get("scheduled_end", ""))
            time_limit_minutes = compute_time_limit_minutes(
                scheduled_start,
                scheduled_end,
                int(request.form.get("time_limit_minutes", "0") or 0),
            )
            monitoring_enabled = request.form.get("monitoring_enabled") == "on"
            questions_payload = json.loads(request.form.get("questions_payload", "[]") or "[]")

            create_or_update_quiz(
                quiz_id=quiz_id or None,
                creator_id=user["id"],
                title=title or "Untitled Quiz",
                description=description,
                subject=subject or "General",
                time_limit_minutes=time_limit_minutes,
                quiz_code=quiz_code,
                monitoring_enabled=monitoring_enabled,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_end,
                status="published" if action == "publish" else "draft",
                questions_payload=questions_payload,
                assigned_section=assigned_section,
            )
            message = "Quiz published successfully." if action == "publish" else "Draft saved successfully."
            return redirect_to("quiz_manager", message=message)

        quiz_id = request.args.get("quizId", "").strip()
        sample_quiz = get_quiz(quiz_id) if quiz_id else None
        sample_quiz = sample_quiz or blank_quiz()
        is_edit_mode = bool(sample_quiz.get("id"))
        return render_template(
            "create_quiz.html",
            sample_quiz=sample_quiz,
            is_edit_mode=is_edit_mode,
            available_sections=available_student_sections(sample_quiz.get("assigned_section", "")),
            format_schedule_input=lambda value: value.replace(" ", "T") if value else "",
        )

    @app.route("/QuizResults")
    @role_required("admin")
    def quiz_results():
        quizzes = get_quizzes()
        if not quizzes:
            return redirect_to("quiz_manager", message="Create a quiz first before viewing results.")
        quiz_id = request.args.get("quizId", quizzes[0]["id"])
        quiz = get_quiz(quiz_id) or quizzes[0]
        results_context = build_quiz_results_context(quiz)
        return render_template(
            "quiz_results.html",
            quiz=quiz,
            **results_context,
        )

    @app.route("/QuizResults/ExportCsv")
    @role_required("admin")
    def export_quiz_results_csv():
        quizzes = get_quizzes()
        if not quizzes:
            return redirect_to("quiz_manager", message="Create a quiz first before exporting results.")

        quiz_id = request.args.get("quizId", quizzes[0]["id"])
        quiz = get_quiz(quiz_id) or quizzes[0]
        results_context = build_quiz_results_context(quiz)

        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow([
            "Rank",
            "Name",
            "Email",
            "Score",
            "Total Points",
            "Percentage",
            "Status",
            "Flag Count",
            "Submission Time",
            "Started At",
        ])
        for attempt in results_context["attempts"]:
            writer.writerow([
                attempt.get("rank") or "",
                attempt.get("student_name", ""),
                attempt.get("student_email", ""),
                int(attempt.get("score", 0) or 0),
                int(attempt.get("total_points", 0) or 0),
                f"{float(attempt.get('percentage', 0) or 0):.2f}",
                attempt_status_label(attempt.get("status")),
                results_context["flags_by_attempt"].get(str(attempt.get("id", "")).strip(), 0),
                attempt.get("submitted_at") or "Still active",
                attempt.get("started_at") or "",
            ])

        return Response(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{quiz_results_csv_filename(quiz)}"',
            },
        )

    @app.route("/ActivityMonitor")
    @role_required("admin")
    def activity_monitor():
        current_time = datetime.now()
        requested_quiz_id = request.args.get("quizId", "").strip()
        live_quiz_id = request.args.get("liveQuizId", "").strip()
        live_mode = request.args.get("live") == "1"
        severity = request.args.get("severity", "all").strip().lower()
        reviewed = request.args.get("reviewed", "all").strip().lower()
        if severity not in {"all", "low", "medium", "high"}:
            severity = "all"
        if reviewed not in {"all", "pending", "reviewed"}:
            reviewed = "all"
        search = request.args.get("student", "").lower().strip()
        selected_student_email = request.args.get("studentEmail", "").strip().lower()
        all_quizzes = get_quizzes()
        reviewable_quizzes = [quiz for quiz in all_quizzes if quiz.get("monitoring_enabled")]
        live_monitorable_quizzes = [quiz for quiz in reviewable_quizzes if is_live_quiz(quiz, current_time)]

        preferred_quiz = next(
            (quiz for quiz in live_monitorable_quizzes if is_live_quiz(quiz, current_time)),
            None,
        )
        if not preferred_quiz:
            preferred_quiz = (
                live_monitorable_quizzes[0]
                if live_monitorable_quizzes
                else (reviewable_quizzes[0] if reviewable_quizzes else (all_quizzes[0] if all_quizzes else None))
            )

        selectable_quizzes = reviewable_quizzes or all_quizzes
        valid_quiz_ids = {quiz["id"] for quiz in selectable_quizzes}
        if requested_quiz_id and requested_quiz_id != "all" and requested_quiz_id not in valid_quiz_ids:
            requested_quiz_id = ""

        quiz_id = requested_quiz_id or (preferred_quiz["id"] if preferred_quiz else "all")
        selected_quiz_record = get_quiz(quiz_id) if quiz_id and quiz_id != "all" else None
        realtime_quizzes = (
            live_monitorable_quizzes
            if quiz_id == "all"
            else ([selected_quiz_record] if is_live_quiz(selected_quiz_record, current_time) else [])
        )
        using_live_student_rows = bool(realtime_quizzes)
        filtered_logs = []
        for quiz in selectable_quizzes:
            filtered_logs.extend(quiz_flags(quiz["id"]))
        if quiz_id and quiz_id != "all":
            filtered_logs = [flag for flag in filtered_logs if flag["quiz_id"] == quiz_id]
        if severity != "all":
            filtered_logs = [flag for flag in filtered_logs if flag["flag_level"] == severity]
        if reviewed != "all":
            expected = reviewed == "reviewed"
            filtered_logs = [flag for flag in filtered_logs if flag["reviewed"] == expected]
        if search:
            filtered_logs = [
                flag for flag in filtered_logs
                if search in flag["student_name"].lower() or search in flag["student_email"].lower()
            ]
        filtered_logs.sort(key=lambda flag: sort_timestamp(flag.get("timestamp")), reverse=True)

        live_student_rows = build_in_progress_student_rows(
            realtime_quizzes,
            severity=severity,
            reviewed=reviewed,
            search=search,
            selected_student_email=selected_student_email,
            now=current_time,
        ) if using_live_student_rows else []

        student_options_map: dict[str, dict] = {}
        for flag in filtered_logs:
            student_email = str(flag.get("student_email", "")).strip().lower()
            student_name = str(flag.get("student_name", "Unknown Student")).strip() or "Unknown Student"
            student_key = student_identity_key(student_name, student_email)
            if student_key not in student_options_map:
                student_options_map[student_key] = {
                    "student_email": str(flag.get("student_email", "")).strip(),
                    "student_name": student_name,
                }
        for row in live_student_rows:
            student_key = row["student_key"]
            student_options_map.setdefault(
                student_key,
                {
                    "student_email": row["student_email"],
                    "student_name": row["student_name"],
                },
            )
        student_options = sorted(
            student_options_map.values(),
            key=lambda item: (item["student_name"].lower(), item["student_email"].lower()),
        )

        if selected_student_email and selected_student_email not in student_options_map:
            selected_student_email = ""

        if selected_student_email:
            filtered_logs = [
                flag for flag in filtered_logs
                if (
                    student_identity_key(
                        str(flag.get("student_name", "Unknown Student")),
                        str(flag.get("student_email", "")),
                    )
                    == selected_student_email
                )
            ]

        quiz_lookup = {quiz["id"]: quiz for quiz in all_quizzes}
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        student_activity_map: dict[str, dict] = {}
        for flag in filtered_logs:
            student_email = str(flag.get("student_email", "")).strip().lower()
            student_name = str(flag.get("student_name", "Unknown Student")).strip() or "Unknown Student"
            student_key = student_email or f"unknown:{student_name.lower()}"
            quiz_title = quiz_lookup.get(flag["quiz_id"], {}).get("title", flag["quiz_id"])
            event_label = activity_event_label(flag.get("event_type", ""))
            entry = student_activity_map.setdefault(
                student_key,
                {
                    "student_name": student_name,
                    "student_email": str(flag.get("student_email", "")).strip(),
                    "event_count": 0,
                    "quiz_ids": set(),
                    "highest_level": "low",
                    "last_activity": "",
                    "latest_event": "",
                    "latest_quiz": "",
                    "reviewed_count": 0,
                    "pending_count": 0,
                    "event_counter": Counter(),
                },
            )
            entry["event_count"] += 1
            entry["quiz_ids"].add(flag["quiz_id"])
            entry["event_counter"][event_label] += 1
            if flag.get("reviewed"):
                entry["reviewed_count"] += 1
            else:
                entry["pending_count"] += 1
            current_rank = severity_rank.get(flag.get("flag_level", "low"), 1)
            saved_rank = severity_rank.get(entry["highest_level"], 1)
            if current_rank >= saved_rank:
                entry["highest_level"] = flag.get("flag_level", "low")
            if not entry["last_activity"]:
                entry["last_activity"] = flag.get("timestamp", "")
                entry["latest_event"] = activity_event_label(flag.get("event_type", ""))
                entry["latest_quiz"] = quiz_title

        historical_student_rows = sorted(
            [
                {
                    **item,
                    "quiz_count": len(item["quiz_ids"]),
                    "event_tallies": item["event_counter"].most_common(3),
                }
                for item in student_activity_map.values()
            ],
            key=lambda item: (
                -severity_rank.get(item["highest_level"], 1),
                -item["event_count"],
                item["student_name"].lower(),
            ),
        )

        student_activity_rows = live_student_rows if using_live_student_rows else historical_student_rows

        selected_student = next(
            (
                row for row in student_activity_rows
                if (
                    student_identity_key(row["student_name"], row["student_email"])
                    == selected_student_email
                )
            ),
            None,
        )
        if not selected_student:
            selected_student = next(
                (
                    row for row in historical_student_rows
                    if student_identity_key(row["student_name"], row["student_email"]) == selected_student_email
                ),
                None,
            )

        valid_live_quiz_ids = {quiz["id"] for quiz in live_monitorable_quizzes}
        if live_quiz_id and live_quiz_id not in valid_live_quiz_ids:
            live_quiz_id = ""

        if not live_quiz_id:
            preferred_live_quiz = (
                get_quiz(quiz_id)
                if quiz_id and quiz_id != "all" and quiz_id in valid_live_quiz_ids
                else (preferred_quiz if preferred_quiz and preferred_quiz["id"] in valid_live_quiz_ids else None)
            )
            if preferred_live_quiz:
                live_quiz_id = preferred_live_quiz["id"]
            else:
                in_progress_quiz = next(
                    (
                        quiz
                        for quiz in live_monitorable_quizzes
                        if any(
                            attempt.get("status") == "in_progress"
                            and not attempt_has_expired(quiz, attempt, current_time)
                            for attempt in quiz_attempts(quiz["id"])
                        )
                    ),
                    None,
                )
                if in_progress_quiz:
                    live_quiz_id = in_progress_quiz["id"]

        live_quiz = get_quiz(live_quiz_id) if live_quiz_id else None

        return render_template(
            "activity_monitor.html",
            stats=build_activity_stats(filtered_logs),
            logs=filtered_logs,
            student_activity_rows=student_activity_rows,
            student_options=student_options,
            selected_student=selected_student,
            selected_student_email=selected_student_email,
            quiz_lookup=quiz_lookup,
            quizzes=selectable_quizzes,
            selected_quiz=quiz_id,
            selected_severity=severity,
            selected_reviewed=reviewed,
            search=search,
            live_mode=live_mode,
            live_quiz=live_quiz,
            live_quiz_id=live_quiz_id,
            realtime_quiz_ids=[quiz["id"] for quiz in realtime_quizzes],
            realtime_enabled=bool(socketio),
        )

    @app.route("/api/quiz/<quiz_id>/in-progress-students")
    @role_required("admin")
    def get_in_progress_students(quiz_id: str):
        """Return all students currently taking a quiz with their activity flags."""
        severity = request.args.get("severity", "all").strip().lower()
        reviewed = request.args.get("reviewed", "all").strip().lower()
        search = request.args.get("student", "").strip().lower()
        selected_student_email = request.args.get("studentEmail", "").strip().lower()
        if severity not in {"all", "low", "medium", "high"}:
            severity = "all"
        if reviewed not in {"all", "pending", "reviewed"}:
            reviewed = "all"

        current_time = datetime.now()
        if quiz_id == "all":
            quizzes = [
                quiz for quiz in get_quizzes()
                if quiz.get("monitoring_enabled") and is_live_quiz(quiz, current_time)
            ]
        else:
            quiz = get_quiz(quiz_id)
            if not quiz:
                return jsonify({"students": [], "error": "Quiz not found"}), 404
            quizzes = [quiz] if is_live_quiz(quiz, current_time) else []

        students = build_in_progress_student_rows(
            quizzes,
            severity=severity,
            reviewed=reviewed,
            search=search,
            selected_student_email=selected_student_email,
            now=current_time,
        )
        return jsonify({"students": students}), 200

    @app.route("/StudentDashboard")
    @role_required("user")
    def student_dashboard():
        user = current_session_user()
        if not user:
            return redirect_to("login")
        current_time = datetime.now()
        summary = student_dashboard_summary(user["email"])
        available_quiz_cards = []
        for quiz in open_quizzes():
            latest_attempt = get_quiz_attempt_for_student(quiz["id"], user["id"])
            if latest_attempt and latest_attempt.get("status") in {"submitted", "auto_submitted"}:
                continue
            access_allowed, _ = quiz_access_state(quiz, user["id"], now=current_time)
            if not access_allowed:
                continue
            available_quiz_cards.append(
                {
                    "quiz": quiz,
                    "window": student_quiz_window_summary(quiz, current_time),
                    "is_in_progress": bool(latest_attempt and latest_attempt.get("status") == "in_progress"),
                }
            )

        available_quiz_cards.sort(
            key=lambda item: (
                parse_schedule(item["quiz"].get("scheduled_end")) or datetime.max,
                parse_schedule(item["quiz"].get("scheduled_start")) or datetime.min,
                item["quiz"].get("title", "").lower(),
            )
        )
        return render_template(
            "student_dashboard.html",
            available_quiz_cards=available_quiz_cards,
            summary=summary,
            user=user,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
            availability_checked_at=format_current_timestamp(current_time),
        )

    @app.route("/StudentCamera")
    @role_required("user")
    def student_camera():
        detection_available, detection_message = get_detection_runtime_status()
        return render_template(
            "student_camera.html",
            detection_available=detection_available,
            detection_message=detection_message,
        )

    @app.route("/detect-face", methods=["POST"])
    @role_required("user")
    def detect_face():
        if cv2 is None or YOLO is None:
            return jsonify({"error": "Detection dependencies are not installed."}), 500

        data = request.get_json(silent=True) or {}
        image_data = str(data.get("image", ""))
        quiz_id = str(data.get("quizId", "")).strip()
        attempt_id = str(data.get("attemptId", "")).strip()
        frame = decode_image_from_data_url(image_data)
        if frame is None:
            return jsonify({"error": "Unable to decode the camera frame."}), 400

        try:
            model = get_detection_model()
            results = model.predict(
                source=frame,
                conf=DETECTION_INFER_CONFIDENCE,
                iou=DETECTION_INFER_IOU,
                imgsz=DETECTION_INFER_IMGSZ,
                max_det=DETECTION_INFER_MAX_DET,
                verbose=False,
            )[0]
            detections = []
            confident_detections = []
            model_has_classification = bool(getattr(model, "names", None))

            def resolve_detection_type(label: str) -> str | None:
                normalized_label = label.strip().lower()
                if normalized_label == "normal" or normalized_label.startswith("normal_"):
                    return "normal"
                if normalized_label == "cheat" or normalized_label == "cheating":
                    return "cheat"
                if normalized_label.startswith("cheat_") or normalized_label.startswith("cheating_"):
                    return "cheat"
                return None
            
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())
                raw_label = str(model.names.get(class_id, class_id))
                detection_type = resolve_detection_type(raw_label)
                if not detection_type:
                    continue

                if confidence < DETECTION_CLASS_DRAW_MIN_CONF:
                    continue

                detection_payload = {
                    "bbox": [coords[0], coords[1], coords[2], coords[3]],
                    "confidence": confidence,
                    "label": raw_label,
                    "raw_label": raw_label,
                    "behavior": detection_type,
                    "type": detection_type,
                }
                detections.append(detection_payload)

                min_class_conf = DETECTION_CLASS_CHEAT_MIN_CONF if detection_type == "cheat" else DETECTION_CLASS_NORMAL_MIN_CONF
                if confidence < min_class_conf:
                    continue

                confident_detections.append(detection_payload)

            if detections and model_has_classification:
                detections.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
            if confident_detections and model_has_classification:
                confident_detections.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
                    
        except Exception as exc:
            return jsonify(
                {
                    "error": f"Detection failed: {exc}",
                    "modelPath": str(DETECTION_MODEL_PATH),
                }
            ), 500

        normal_count = sum(1 for item in detections if item.get("type") == "normal")
        cheating_count = len(detections) - normal_count
        result_state = "normal"
        result_message = "normal"
        flag_level = "low"
        event_type = "normal"
        top_detection = confident_detections[0] if confident_detections else (detections[0] if detections else None)
        top_cheat_conf = max((float(item.get("confidence", 0)) for item in confident_detections if item.get("type") == "cheat"), default=0.0)
        top_normal_conf = max((float(item.get("confidence", 0)) for item in confident_detections if item.get("type") == "normal"), default=0.0)
        confident_cheat_count = sum(1 for item in confident_detections if item.get("type") == "cheat")
        confident_normal_count = sum(1 for item in confident_detections if item.get("type") == "normal")
        selected_detection = top_detection

        cheat_is_dominant = (
            top_cheat_conf >= DETECTION_CLASS_CHEAT_STRICT_MIN_CONF
            and top_cheat_conf >= (top_normal_conf + DETECTION_CLASS_MARGIN)
            and confident_cheat_count >= max(1, confident_normal_count)
        )

        if not detections:
            result_state = "normal"
            result_message = "normal"
            flag_level = "low"
        elif not confident_detections:
            result_state = "normal"
            result_message = "normal"
            flag_level = "low"
        elif cheat_is_dominant:
            result_state = "cheat"
            result_message = "cheat"
            flag_level = "high"
            event_type = "cheat"
        else:
            result_state = "normal"
            result_message = "normal"
            selected_detection = next((item for item in confident_detections if item.get("type") == "normal"), top_detection)

        detection_status = {
            "state": result_state,
            "reason": None,
            "message": result_message,
            "normal_count": normal_count,
            "suspicious_count": cheating_count,
            "model_label": (selected_detection or {}).get("raw_label", (selected_detection or {}).get("label", "")),
            "behavior_label": (selected_detection or {}).get("behavior", (selected_detection or {}).get("type", "")),
            "model_confidence": float((selected_detection or {}).get("confidence", 0.0)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "flag_level": flag_level,
        }

        user = current_session_user()
        attempt = get_attempt(attempt_id) if attempt_id else None
        valid_attempt = bool(
            attempt
            and quiz_id
            and attempt.get("quiz_id") == quiz_id
            and user
            and attempt.get("student_id") == user.get("id")
            and attempt.get("status") == "in_progress"
        )

        created_log = None
        if valid_attempt and event_type == "cheat" and _should_log_detection_event(attempt_id, event_type):
            created_log = create_activity_log(
                quiz_id=quiz_id,
                attempt_id=attempt_id,
                event_type=event_type,
                event_description=result_message,
                flag_level=flag_level,
            )
            if socketio:
                socketio.emit(
                    "activity_log_created",
                    {
                        "quizId": quiz_id,
                        "log": activity_log_with_details(created_log),
                    },
                    to=f"monitor:{quiz_id}",
                )

        return jsonify({"detections": detections, "detection": detection_status, "logged": bool(created_log)})

    @app.route("/JoinQuiz")
    @role_required("user")
    def join_quiz():
        code = request.args.get("code", "").strip().upper()
        quiz = get_quiz_by_code(code) if code else None
        access_allowed = False
        access_message = None
        lookup_message = ""
        user = current_session_user()
        if not user:
            return redirect_to("login")
        if quiz:
            access_allowed, access_message = quiz_access_state(quiz, user["id"])
        elif code:
            lookup_message = "No quiz was found for that access code. Double-check the code and try again."
        return render_template(
            "join_quiz.html",
            quiz=quiz,
            code=code,
            access_allowed=access_allowed,
            access_message=access_message,
            lookup_message=lookup_message,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
        )

    @app.route("/TakeQuiz", methods=["GET", "POST"])
    @role_required("user")
    def take_quiz():
        quiz_id = request.values.get("quizId", "").strip()
        quiz = get_quiz(quiz_id) if quiz_id else None
        if not quiz:
            return redirect_to("student_dashboard")
        user = current_session_user()
        if not user:
            return redirect_to("login")
        current_time = datetime.now()
        latest_attempt = get_quiz_attempt_for_student(quiz["id"], user["id"])
        submitted = request.args.get("submitted") == "1"
        attempt = None
        active_attempt_id = ""
        countdown_seconds = None
        submission_error = ""
        submitted_answers: dict[str, str] = {}

        if submitted:
            submitted_attempt = get_attempt(request.args.get("attemptId", "").strip())
            if submitted_attempt and submitted_attempt.get("quiz_id") == quiz["id"] and submitted_attempt.get("student_id") == user["id"]:
                attempt = submitted_attempt
            else:
                submitted = False

        if request.method == "POST" and not submitted:
            attempt_id = request.form.get("attempt_id", "").strip()
            if attempt_id:
                attempt = get_attempt(attempt_id)
                valid_owner = bool(attempt and attempt.get("quiz_id") == quiz["id"] and attempt.get("student_id") == user["id"])
                if not valid_owner:
                    attempt = None
                    attempt_id = ""
            if not attempt_id and latest_attempt and latest_attempt.get("status") == "in_progress":
                attempt = latest_attempt
                attempt_id = latest_attempt["id"]

            access_allowed, access_message = quiz_access_state(quiz, user["id"], now=current_time)
            if attempt_id or access_allowed:
                if not attempt_id:
                    attempt_id = ensure_quiz_attempt_in_progress(quiz["id"], user["id"], False)
                    attempt = get_attempt(attempt_id)
                answers = {
                    question["id"]: request.form.get(f"question_{question['id']}", "").strip()
                    for question in quiz["questions"]
                }
                submitted_answers = dict(answers)
                consent_given = request.form.get("consent_given") == "on" or not quiz["monitoring_enabled"]
                is_overdue_submission = attempt_has_expired(quiz, attempt, current_time)
                if not is_overdue_submission:
                    unanswered_numbers = [
                        index
                        for index, question in enumerate(quiz["questions"], start=1)
                        if not answers.get(question["id"], "")
                    ]
                    if unanswered_numbers:
                        label = "question" if len(unanswered_numbers) == 1 else "questions"
                        submission_error = (
                            f"Answer all questions before submitting. Missing {label}: "
                            f"{', '.join(str(number) for number in unanswered_numbers)}."
                        )
                        active_attempt_id = attempt_id
                        countdown_seconds = remaining_attempt_seconds(quiz, attempt, current_time)
                    else:
                        attempt_id = finalize_quiz_attempt(attempt_id, answers, consent_given, status="submitted")
                        return redirect_to("take_quiz", quizId=quiz["id"], submitted=1, attemptId=attempt_id)
                else:
                    attempt_id = finalize_quiz_attempt(attempt_id, answers, consent_given, status="auto_submitted")
                    return redirect_to("take_quiz", quizId=quiz["id"], submitted=1, attemptId=attempt_id)

        if not submitted and latest_attempt and latest_attempt.get("status") == "in_progress" and attempt_has_expired(quiz, latest_attempt, current_time):
            expired_attempt_id = finalize_quiz_attempt(
                latest_attempt["id"],
                {},
                bool(latest_attempt.get("consent_given")),
                status="auto_submitted",
            )
            return redirect_to("take_quiz", quizId=quiz["id"], submitted=1, attemptId=expired_attempt_id)

        access_allowed, access_message = quiz_access_state(quiz, user["id"], now=current_time)
        if submitted and attempt:
            access_allowed = True
            access_message = None

        if request.method == "GET" and access_allowed and not submitted:
            active_attempt_id = ensure_quiz_attempt_in_progress(quiz["id"], user["id"], False)
            active_attempt = get_attempt(active_attempt_id)
            countdown_seconds = remaining_attempt_seconds(quiz, active_attempt, current_time)
            if countdown_seconds is not None and countdown_seconds <= 0:
                expired_attempt_id = finalize_quiz_attempt(
                    active_attempt_id,
                    {},
                    bool((active_attempt or {}).get("consent_given")),
                    status="auto_submitted",
                )
                return redirect_to("take_quiz", quizId=quiz["id"], submitted=1, attemptId=expired_attempt_id)

        detection_available, detection_message = get_detection_runtime_status()

        return render_template(
            "take_quiz.html",
            quiz=quiz,
            attempt=attempt,
            submitted=submitted and bool(attempt),
            access_allowed=access_allowed,
            access_message=access_message,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
            realtime_enabled=bool(socketio),
            current_user_name=(user or {}).get("full_name", "Student"),
            active_attempt_id=active_attempt_id,
            remaining_attempt_seconds=countdown_seconds,
            detection_available=detection_available,
            detection_message=detection_message,
            submission_error=submission_error,
            submitted_answers=submitted_answers,
        )

    @app.route("/UserManagement")
    @role_required("admin")
    def user_management():
        current_user = current_session_user() or {}
        message = request.args.get("message", "").strip()
        error = request.args.get("error", "").strip()
        relationship_counts = user_record_counts()
        users = []
        for user in get_users():
            counts = relationship_counts.get(user["id"], {"owned_quizzes": 0, "quiz_attempts": 0})
            can_delete = (
                user["id"] != current_user.get("id")
                and counts.get("owned_quizzes", 0) == 0
                and counts.get("quiz_attempts", 0) == 0
            )
            users.append(
                {
                    **user,
                    "avatar_url": _normalize_avatar_url(user.get("avatar_url")),
                    "owned_quizzes": counts.get("owned_quizzes", 0),
                    "quiz_attempts": counts.get("quiz_attempts", 0),
                    "can_delete": can_delete,
                    "is_current_user": user["id"] == current_user.get("id"),
                }
            )

        student_count = sum(1 for user in users if user["role"] == "user")
        instructor_count = sum(1 for user in users if user["role"] == "admin")
        return render_template(
            "user_management.html",
            users=users,
            total_users=len(users),
            student_count=student_count,
            instructor_count=instructor_count,
            message=message,
            error=error,
        )

    @app.route("/UserManagement/Save", methods=["POST"])
    @role_required("admin")
    def save_user_management_user():
        current_user = current_session_user() or {}
        user_id = request.form.get("user_id", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "").strip().lower()
        password = "" if user_id else request.form.get("password", "")
        section_name = request.form.get("section_name", "").strip()

        if not full_name:
            return redirect_to("user_management", error="Full name is required.")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return redirect_to("user_management", error="Enter a valid email address.")
        if role not in {"admin", "user"}:
            return redirect_to("user_management", error="Choose a valid role.")
        if not user_id and len(password) < 8:
            return redirect_to("user_management", error="New users need a password with at least 8 characters.")
        if user_id and current_user.get("id") == user_id and role != "admin":
            return redirect_to("user_management", error="You cannot remove your own instructor access while signed in.")

        try:
            if user_id:
                update_user(user_id, email, full_name, role, None, section_name=section_name)
                message = "User updated successfully."
            else:
                create_user(email, full_name, role, password, section_name=section_name)
                message = "User created successfully."
        except ValueError as exc:
            return redirect_to("user_management", error=str(exc))
        except sqlite3.IntegrityError:
            return redirect_to("user_management", error="This email is already in use.")

        return redirect_to("user_management", message=message)

    @app.route("/UserManagement/Delete", methods=["POST"])
    @role_required("admin")
    def delete_user_management_user():
        current_user = current_session_user() or {}
        user_id = request.form.get("user_id", "").strip()
        user = get_user_by_id(user_id)
        if not user:
            return redirect_to("user_management", error="User not found.")
        if user_id == current_user.get("id"):
            return redirect_to("user_management", error="You cannot delete the account you are currently using.")

        counts = user_record_counts().get(user_id, {"owned_quizzes": 0, "quiz_attempts": 0})
        if counts.get("owned_quizzes", 0) or counts.get("quiz_attempts", 0):
            return redirect_to(
                "user_management",
                error="This user cannot be deleted because they still own quizzes or have quiz attempt records.",
            )

        if delete_user_by_id(user_id):
            return redirect_to("user_management", message="User deleted successfully.")
        return redirect_to("user_management", error="User could not be deleted.")

    if socketio:
        def room_name(quiz_id: str) -> str:
            return f"monitor:{quiz_id}"

        def room_participants(quiz_id: str) -> dict[str, dict]:
            return _monitor_rooms.setdefault(room_name(quiz_id), {})

        def validate_monitor_room_access(quiz_id: str, role: str, user: dict | None, attempt_id: str = "") -> tuple[bool, str, dict | None]:
            quiz = get_quiz(quiz_id)
            if not quiz:
                return False, "Quiz not found.", None
            if not quiz.get("monitoring_enabled"):
                return False, "Live monitoring is not enabled for this quiz.", None
            if role == "admin":
                return True, "", None
            if role != "user" or not user:
                return False, "Unauthorized realtime connection.", None
            if not attempt_id:
                return False, "Missing active quiz attempt.", None

            attempt = get_attempt(attempt_id)
            valid_attempt = bool(
                attempt
                and attempt.get("quiz_id") == quiz_id
                and attempt.get("student_id") == user.get("id")
                and attempt.get("status") == "in_progress"
            )
            if not valid_attempt:
                return False, "You do not have an active quiz session for this room.", None
            return True, "", attempt

        def participant_payload(participant: dict) -> dict:
            return {
                "sid": participant.get("sid", ""),
                "quiz_id": participant.get("quiz_id", ""),
                "role": participant.get("role", ""),
                "user_id": participant.get("user_id", ""),
                "attempt_id": participant.get("attempt_id", ""),
                "display_name": participant.get("display_name", ""),
                "email": participant.get("email", ""),
                "camera_on": bool(participant.get("camera_on", False)),
            }

        @socketio.on("join_monitor_room")
        def on_join_monitor_room(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            if not quiz_id:
                emit("monitor_error", {"message": "Missing quiz ID."})
                return

            current_user = current_session_user()
            role = (current_user or {}).get("role", "")
            if role not in {"admin", "user"}:
                emit("monitor_error", {"message": "Unauthorized realtime connection."})
                return

            raw_attempt_id = str((data or {}).get("attemptId", "")).strip()
            access_allowed, access_message, active_attempt = validate_monitor_room_access(
                quiz_id,
                role,
                current_user,
                raw_attempt_id,
            )
            if not access_allowed:
                emit("monitor_error", {"message": access_message})
                return

            room = room_name(quiz_id)
            join_room(room)

            participants = room_participants(quiz_id)
            sid = request.sid
            display_name = ((current_user or {}).get("full_name") or role.title()).strip()
            attempt_id = active_attempt.get("id", "") if active_attempt else ""
            user_id = (current_user or {}).get("id", "")
            participants[sid] = {
                "sid": sid,
                "quiz_id": quiz_id,
                "role": role,
                "user_id": user_id,
                "attempt_id": attempt_id,
                "display_name": display_name,
                "email": (current_user or {}).get("email", ""),
                "camera_on": bool((data or {}).get("cameraOn", False) and role == "user" and active_attempt),
            }

            emit(
                "room_snapshot",
                {
                    "quizId": quiz_id,
                    "participants": [participant_payload(item) for item in participants.values()],
                },
                to=sid,
            )
            emit("participant_joined", participant_payload(participants[sid]), room=room, include_self=False)

        @socketio.on("set_camera_status")
        def on_set_camera_status(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            camera_on = bool((data or {}).get("cameraOn", False))
            if not quiz_id:
                return

            participants = room_participants(quiz_id)
            participant = participants.get(request.sid)
            if not participant:
                return

            if participant.get("role") != "user":
                camera_on = False

            participant["camera_on"] = camera_on
            emit("participant_updated", participant_payload(participant), room=room_name(quiz_id))

        @socketio.on("webrtc_offer")
        def on_webrtc_offer(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            target_sid = str((data or {}).get("targetSid", "")).strip()
            description = (data or {}).get("description")
            if not quiz_id or not target_sid or not description:
                return
            participants = room_participants(quiz_id)
            if request.sid not in participants or target_sid not in participants:
                return
            sender = participants.get(request.sid, {})
            target = participants.get(target_sid, {})
            if sender.get("role") != "admin" or target.get("role") != "user":
                return
            emit(
                "webrtc_offer",
                {
                    "quizId": quiz_id,
                    "senderSid": request.sid,
                    "senderName": sender.get("display_name", ""),
                    "description": description,
                },
                to=target_sid,
            )

        @socketio.on("webrtc_answer")
        def on_webrtc_answer(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            target_sid = str((data or {}).get("targetSid", "")).strip()
            description = (data or {}).get("description")
            if not quiz_id or not target_sid or not description:
                return
            participants = room_participants(quiz_id)
            if request.sid not in participants or target_sid not in participants:
                return
            sender = participants.get(request.sid, {})
            target = participants.get(target_sid, {})
            if sender.get("role") != "user" or target.get("role") != "admin":
                return
            emit(
                "webrtc_answer",
                {
                    "quizId": quiz_id,
                    "senderSid": request.sid,
                    "description": description,
                },
                to=target_sid,
            )

        @socketio.on("webrtc_ice_candidate")
        def on_webrtc_ice_candidate(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            target_sid = str((data or {}).get("targetSid", "")).strip()
            candidate = (data or {}).get("candidate")
            if not quiz_id or not target_sid or not candidate:
                return
            participants = room_participants(quiz_id)
            if request.sid not in participants or target_sid not in participants:
                return
            sender = participants.get(request.sid, {})
            target = participants.get(target_sid, {})
            if {sender.get("role"), target.get("role")} != {"admin", "user"}:
                return
            emit(
                "webrtc_ice_candidate",
                {
                    "quizId": quiz_id,
                    "senderSid": request.sid,
                    "candidate": candidate,
                },
                to=target_sid,
            )

        @socketio.on("monitor_status_report")
        def on_monitor_status_report(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            if not quiz_id:
                return

            participants = room_participants(quiz_id)
            participant = participants.get(request.sid)
            if not participant or participant.get("role") != "user":
                return

            attempt_id = str(participant.get("attempt_id", "")).strip() or str((data or {}).get("attemptId", "")).strip()
            message = str((data or {}).get("message", "")).strip()
            code = str((data or {}).get("code", "")).strip().lower() or "camera_status"
            level = str((data or {}).get("level", "")).strip().lower() or "medium"
            if level not in {"low", "medium", "high"}:
                level = "medium"
            if not message:
                return

            payload = {
                "quizId": quiz_id,
                "attemptId": attempt_id,
                "studentName": participant.get("display_name", "Student"),
                "studentEmail": participant.get("email", ""),
                "message": message,
                "code": code,
                "level": level,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            emit("monitor_status_report", payload, room=room_name(quiz_id))

            attempt = get_attempt(attempt_id) if attempt_id else None
            valid_attempt = bool(
                attempt
                and attempt.get("quiz_id") == quiz_id
                and attempt.get("student_id") == participant.get("user_id")
            )
            if valid_attempt and _should_log_detection_event(attempt_id, f"monitor_status:{code}", cooldown_seconds=15):
                created_log = create_activity_log(
                    quiz_id=quiz_id,
                    attempt_id=attempt_id,
                    event_type=code,
                    event_description=message,
                    flag_level=level,
                )
                emit(
                    "activity_log_created",
                    {
                        "quizId": quiz_id,
                        "log": activity_log_with_details(created_log),
                    },
                    to=room_name(quiz_id),
                )

        @socketio.on("disconnect")
        def on_disconnect():
            sid = request.sid
            for room, participants in list(_monitor_rooms.items()):
                if sid not in participants:
                    continue
                participant = participants.pop(sid)
                leave_room(room)
                emit("participant_left", {"sid": sid, "quizId": participant.get("quiz_id", "")}, room=room)
                if not participants:
                    _monitor_rooms.pop(room, None)

    return app
