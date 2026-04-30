from __future__ import annotations

from pathlib import Path

import base64
import json
import os
import re
import secrets
import ssl
import smtplib
from email.message import EmailMessage
from io import BytesIO
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

from .data import (
    activity_log_with_details,
    activity_stats,
    build_dashboard_calendar,
    cheating_summary,
    create_activity_log,
    create_or_update_quiz,
    create_user,
    dashboard_stats,
    delete_quiz_by_id,
    delete_user_by_email,
    ensure_quiz_attempt_in_progress,
    finalize_quiz_attempt,
    format_schedule,
    get_attempt,
    get_quiz,
    get_quiz_by_code,
    get_quizzes,
    get_user,
    get_user_by_email,
    get_user_by_id,
    get_users,
    init_database,
    open_quizzes,
    quiz_attempts,
    quiz_access_state,
    quiz_flags,
    parse_schedule,
    schedule_status,
    student_dashboard_summary,
    student_attempts,
    submit_quiz_attempt,
    set_user_password,
    verify_password,
    set_quiz_status,
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

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading") if SocketIO else None
_monitor_rooms: dict[str, dict[str, dict]] = {}
_detection_event_cache: dict[str, datetime] = {}
_password_reset_code_store: dict[str, dict] = {}
_shared_session: dict[str, str] = {}  # Stores shared session: {"user_id": "", "role": ""}


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
    app.config["SECRET_KEY"] = "semcds-demo-secret"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
    app.config["PREFERRED_URL_SCHEME"] = "https"  # Use https by default for ngrok
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_ENV", "development") == "production"
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Allow cross-site access
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    init_database()

    if socketio:
        socketio.init_app(app, manage_session=False, cors_allowed_origins="*")

    def redirect_to(endpoint, **kwargs):
        """Redirect to an endpoint using relative URLs to preserve the current host.
        This ensures redirects work correctly whether accessed via localhost or ngrok URL."""
        return redirect(url_for(endpoint, **kwargs, _external=False))

    @app.before_request
    def auto_login_from_shared_session():
        """Automatically login if shared session exists and user is not already logged in."""
        if "user_id" not in session and _shared_session.get("user_id"):
            session["user_id"] = _shared_session["user_id"]
            session["role"] = _shared_session["role"]
            session.permanent = True

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

    def _current_external_callback_uri() -> str:
        """Build callback URI from forwarded headers so ngrok devices never redirect to localhost."""
        forwarded_proto = (request.headers.get("X-Forwarded-Proto", "") or "").split(",")[0].strip().lower()
        forwarded_host = (request.headers.get("X-Forwarded-Host", "") or "").split(",")[0].strip()

        scheme = forwarded_proto or request.scheme or "https"
        host = forwarded_host or request.host

        # Normalize default ports to avoid redirect_uri mismatch.
        if scheme == "https" and host.endswith(":443"):
            host = host[:-4]
        elif scheme == "http" and host.endswith(":80"):
            host = host[:-3]

        callback_path = url_for("google_callback", _external=False)
        return f"{scheme}://{host}{callback_path}"

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
        state = role if role in {"admin", "user"} else "user"
        query = urllib_parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "prompt": "select_account",
                "access_type": "offline",
                "state": state,
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

    def _send_invitation_email(recipient_email: str, temporary_password: str | None = None) -> None:
        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@example.com").strip()
        if not smtp_host or not smtp_user or not smtp_password:
            raise RuntimeError("Email sending is not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD.")

        login_url = url_for("login", _external=True)
        body_lines = [
            f"Hello,\n\nYou've been invited to join SEMCDS as a Student.",
            "\n",
        ]
        if temporary_password:
            body_lines.append(
                f"A new account has been created for you.\n\nEmail: {recipient_email}\nTemporary password: {temporary_password}\n"
            )
        body_lines.append(f"Sign in here: {login_url}\n\nIf you did not expect this invitation, please ignore this message.")
        message = EmailMessage()

        effective_sender = smtp_from
        if "gmail.com" in smtp_host.lower() or "googlemail.com" in smtp_host.lower():
            effective_sender = smtp_user
            if smtp_from.lower() != smtp_user.lower():
                message["Reply-To"] = smtp_from

        message["From"] = effective_sender
        message["To"] = recipient_email
        message["Subject"] = "You're invited to SEMCDS"
        message.set_content("\n".join(body_lines))

        context = ssl.create_default_context()
        if smtp_port == 465:
            smtp_client = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20, context=context)
        else:
            smtp_client = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        try:
            if smtp_port != 465:
                smtp_client.ehlo()
                smtp_client.starttls(context=context)
                smtp_client.ehlo()
            smtp_client.login(smtp_user, smtp_password)
            smtp_client.send_message(message, from_addr=effective_sender)
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
            raise RuntimeError(f"SMTP error while sending invitation: {exc}") from exc
        finally:
            smtp_client.quit()

    def _send_password_reset_email(recipient_email: str, reset_url: str) -> None:
        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@example.com").strip()
        if not smtp_host or not smtp_user or not smtp_password:
            raise RuntimeError("Email sending is not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD.")

        message = EmailMessage()

        effective_sender = smtp_from
        if "gmail.com" in smtp_host.lower() or "googlemail.com" in smtp_host.lower():
            effective_sender = smtp_user
            if smtp_from.lower() != smtp_user.lower():
                message["Reply-To"] = smtp_from

        message["From"] = effective_sender
        message["To"] = recipient_email
        message["Subject"] = "SEMCDS password reset"
        message.set_content(
            "\n".join(
                [
                    "Hello,",
                    "",
                    "We received a request to reset your SEMCDS password.",
                    f"Reset your password using this link: {reset_url}",
                    "",
                    "This link expires in 60 minutes.",
                    "If you did not request a reset, you can ignore this email.",
                ]
            )
        )

        context = ssl.create_default_context()
        if smtp_port == 465:
            smtp_client = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20, context=context)
        else:
            smtp_client = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        try:
            if smtp_port != 465:
                smtp_client.ehlo()
                smtp_client.starttls(context=context)
                smtp_client.ehlo()
            smtp_client.login(smtp_user, smtp_password)
            smtp_client.send_message(message, from_addr=effective_sender)
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
            raise RuntimeError(f"SMTP error while sending password reset email: {exc}") from exc
        finally:
            smtp_client.quit()

    def _send_password_reset_code_email(recipient_email: str, reset_code: str) -> None:
        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@example.com").strip()
        if not smtp_host or not smtp_user or not smtp_password:
            raise RuntimeError("Email sending is not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD.")

        message = EmailMessage()

        effective_sender = smtp_from
        if "gmail.com" in smtp_host.lower() or "googlemail.com" in smtp_host.lower():
            effective_sender = smtp_user
            if smtp_from.lower() != smtp_user.lower():
                message["Reply-To"] = smtp_from

        message["From"] = effective_sender
        message["To"] = recipient_email
        message["Subject"] = "SEMCDS password reset code"
        message.set_content(
            "\n".join(
                [
                    "Hello,",
                    "",
                    "We received a request to reset your SEMCDS password.",
                    f"Your 6-digit reset code is: {reset_code}",
                    "",
                    "This code expires in 10 minutes.",
                    "If you did not request a reset, you can ignore this email.",
                ]
            )
        )

        context = ssl.create_default_context()
        if smtp_port == 465:
            smtp_client = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20, context=context)
        else:
            smtp_client = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        try:
            if smtp_port != 465:
                smtp_client.ehlo()
                smtp_client.starttls(context=context)
                smtp_client.ehlo()
            smtp_client.login(smtp_user, smtp_password)
            smtp_client.send_message(message, from_addr=effective_sender)
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
            raise RuntimeError(f"SMTP error while sending password reset code email: {exc}") from exc
        finally:
            smtp_client.quit()

    def _purge_stale_password_reset_codes(ttl_seconds: int) -> None:
        cutoff = datetime.now() - timedelta(seconds=ttl_seconds * 2)
        stale_emails = []
        for email, payload in _password_reset_code_store.items():
            sent_at = payload.get("sent_at")
            if not isinstance(sent_at, datetime):
                stale_emails.append(email)
                continue
            if sent_at < cutoff:
                stale_emails.append(email)
        for email in stale_emails:
            _password_reset_code_store.pop(email, None)

    def _build_password_reset_token(user: dict) -> str:
        return password_reset_serializer.dumps(
            {"user_id": user.get("id", ""), "email": user.get("email", "")},
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
        if not user_id or not email:
            return None
        return {"user_id": user_id, "email": email}

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

    def compute_time_limit_minutes(scheduled_start: str, scheduled_end: str, fallback_minutes: int = 15) -> int:
        start = parse_schedule(scheduled_start)
        end = parse_schedule(scheduled_end)
        if start and end and end > start:
            minutes = int((end - start).total_seconds() // 60)
            return max(1, minutes)
        return max(1, fallback_minutes)

    def blank_quiz() -> dict:
        return {
            "id": "",
            "title": "",
            "description": "",
            "subject": "",
            "time_limit_minutes": 15,
            "status": "draft",
            "quiz_code": "",
            "monitoring_enabled": False,
            "scheduled_start": "",
            "scheduled_end": "",
            "questions": [],
            "total_points": 0,
        }

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
        role = session.get("role")
        current_user = get_user_by_id(session.get("user_id", "")) if session.get("user_id") else None
        return {
            "current_role": role,
            "current_user": current_user or (get_user(role) if role else None),
            "current_endpoint": request.endpoint,
        }

    def role_required(*allowed_roles):
        def decorator(view):
            @wraps(view)
            def wrapped(*args, **kwargs):
                role = session.get("role")
                if not role:
                    return redirect_to("login")
                if role not in allowed_roles:
                    return redirect_to("home")
                return view(*args, **kwargs)

            return wrapped

        return decorator

    @app.route("/")
    def index():
        if session.get("role") in {"admin", "user"}:
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
                session["role"] = user["role"]
                session["user_id"] = user["id"]
                session.permanent = remember_me
                # Store in shared session for other PCs on same ngrok URL
                _shared_session["user_id"] = user["id"]
                _shared_session["role"] = user["role"]
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
        role = request.args.get("state", "user").strip()
        if role not in {"admin", "user"}:
            role = "user"
        if not code:
            return redirect_to("login", role=role, error="Google login failed.")
        try:
            profile = _fetch_google_user_info(code)
        except Exception as exc:
            return redirect_to("login", role=role, error=str(exc))

        email = (profile.get("email") or "").strip().lower()
        full_name = (profile.get("name") or profile.get("given_name") or email).strip()
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

        session["role"] = role
        session["user_id"] = user["id"]
        _shared_session["user_id"] = user["id"]
        _shared_session["role"] = role
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
        resend_remaining_seconds = 0
        password_reset_code_ttl_seconds = int(os.environ.get("PASSWORD_RESET_CODE_TTL_SECONDS", "600"))
        resend_cooldown_seconds = 60

        _purge_stale_password_reset_codes(password_reset_code_ttl_seconds)

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            action = request.form.get("action", "send_code").strip().lower()
            email_value = email
            user = get_user_by_email(email)

            reset_record = _password_reset_code_store.get(email)
            now = datetime.now()

            if action == "send_code":
                if user and reset_record and isinstance(reset_record.get("sent_at"), datetime):
                    elapsed_seconds = int((now - reset_record["sent_at"]).total_seconds())
                    if elapsed_seconds < resend_cooldown_seconds:
                        resend_remaining_seconds = resend_cooldown_seconds - elapsed_seconds
                        error = f"Please wait {resend_remaining_seconds}s before requesting a new code."

                if not error and user:
                    reset_code = f"{secrets.randbelow(1000000):06d}"
                    _password_reset_code_store[email] = {
                        "code": reset_code,
                        "user_id": user["id"],
                        "sent_at": now,
                        "expires_at": now + timedelta(seconds=password_reset_code_ttl_seconds),
                    }
                    try:
                        _send_password_reset_code_email(user["email"], reset_code)
                    except RuntimeError as exc:
                        error = str(exc)
                    except Exception as exc:
                        error = f"Unable to send reset code email right now. ({exc})"

                if not error:
                    message = "If the email exists in the system, a 6-digit reset code was sent."

            else:
                code = request.form.get("code", "").strip()
                password = request.form.get("password", "")
                confirm_password = request.form.get("confirm_password", "")

                if not email or not code or not password or not confirm_password:
                    error = "Please complete all fields to reset your password."
                elif len(code) != 6 or not code.isdigit():
                    error = "Enter a valid 6-digit reset code."
                elif not user:
                    error = "Invalid email or reset code."
                elif not reset_record:
                    error = "Reset code not found or expired. Request a new code."
                elif reset_record.get("user_id") != user.get("id"):
                    error = "Invalid reset request. Please request a new code."
                elif not isinstance(reset_record.get("expires_at"), datetime) or now > reset_record["expires_at"]:
                    _password_reset_code_store.pop(email, None)
                    error = "Reset code has expired. Request a new code."
                elif str(reset_record.get("code", "")) != code:
                    error = "Invalid reset code."
                elif len(password) < 8:
                    error = "Password must be at least 8 characters long."
                elif password != confirm_password:
                    error = "Passwords do not match."
                elif not set_user_password(user["id"], password):
                    error = "Unable to reset password right now. Please try again."
                else:
                    _password_reset_code_store.pop(email, None)
                    return redirect(
                        url_for(
                            "login",
                            message="Password reset successful. You can now sign in with your new password.",
                            role=user.get("role", "admin"),
                        )
                    )

            active_record = _password_reset_code_store.get(email)
            if user and active_record and isinstance(active_record.get("sent_at"), datetime):
                elapsed_seconds = int((datetime.now() - active_record["sent_at"]).total_seconds())
                if elapsed_seconds < resend_cooldown_seconds:
                    resend_remaining_seconds = max(resend_cooldown_seconds - elapsed_seconds, 0)

        return render_template(
            "forgot_password.html",
            message=message,
            error=error,
            email_value=email_value,
            resend_remaining_seconds=resend_remaining_seconds,
        )

    @app.route("/reset-password/<token>", methods=["GET", "POST"])
    def reset_password(token: str):
        token_payload = _read_password_reset_token(token)
        if not token_payload:
            return render_template(
                "reset_password.html",
                token_valid=False,
                error="This password reset link is invalid or has expired.",
                success_message="",
                token=token,
            )

        user = get_user_by_id(token_payload["user_id"])
        if not user or user.get("email", "").strip().lower() != token_payload["email"]:
            return render_template(
                "reset_password.html",
                token_valid=False,
                error="This password reset link is invalid or has expired.",
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

    @app.route("/signin/<role>")
    def signin(role: str):
        if role not in {"admin", "user"}:
            return redirect_to("login")
        user = get_user(role)
        session["role"] = role
        session["user_id"] = user["id"]
        _shared_session["user_id"] = user["id"]
        _shared_session["role"] = role
        return redirect_to("home")

    @app.route("/logout")
    def logout():
        session.clear()
        # Clear shared session when logging out
        _shared_session.clear()
        return redirect(url_for("login", logout_success="1"))

    @app.route("/api/shared-session", methods=["GET", "POST", "DELETE"])
    def manage_shared_session():
        """API to manage shared session for ngrok multi-device access."""
        if request.method == "GET":
            # Check if shared session is active
            return jsonify({
                "active": bool(_shared_session.get("user_id")),
                "user_id": _shared_session.get("user_id", ""),
                "role": _shared_session.get("role", ""),
            })
        elif request.method == "DELETE":
            # Clear shared session
            _shared_session.clear()
            session.clear()
            return jsonify({"success": True, "message": "Shared session cleared"})
        return jsonify({"error": "Invalid method"}), 405

    @app.route("/home")
    def home():
        role = session.get("role")
        if role == "admin":
            return redirect_to("dashboard")
        if role == "user":
            return redirect_to("student_dashboard")
        return redirect_to("login")

    @app.route("/Dashboard")
    @role_required("admin")
    def dashboard():
        selected_quiz = request.args.get("quizId", "").strip()
        analyzed = request.args.get("analyzed") == "1"
        all_quizzes = get_quizzes()
        analyzable_quizzes = [quiz for quiz in all_quizzes if quiz["status"] == "published" and quiz["monitoring_enabled"]]
        valid_quiz_ids = {quiz["id"] for quiz in analyzable_quizzes}
        if selected_quiz not in valid_quiz_ids:
            selected_quiz = ""
            analyzed = False
        month_param = request.args.get("month", datetime.now().strftime("%Y-%m"))
        selected_day = request.args.get("day", "").strip()
        try:
            current_month = datetime.strptime(month_param, "%Y-%m")
        except ValueError:
            current_month = datetime.now().replace(day=1)
        calendar_rows = build_dashboard_calendar(current_month.year, current_month.month)
        selected_day_quizzes = []
        if selected_day:
            for week in calendar_rows:
                for day in week:
                    if day["key"] == selected_day:
                        selected_day_quizzes = day["quizzes"]
                        break
        prev_month = current_month.month - 1 or 12
        prev_year = current_month.year - 1 if current_month.month == 1 else current_month.year
        next_month = 1 if current_month.month == 12 else current_month.month + 1
        next_year = current_month.year + 1 if current_month.month == 12 else current_month.year
        return render_template(
            "dashboard.html",
            stats=dashboard_stats(),
            recent_quizzes=all_quizzes[:4],
            summary=cheating_summary(selected_quiz) if selected_quiz and analyzed else None,
            quizzes=analyzable_quizzes,
            attempts=[],
            selected_quiz=selected_quiz,
            analyzed=analyzed,
            calendar_rows=calendar_rows,
            calendar_month=current_month.strftime("%B %Y"),
            calendar_month_key=current_month.strftime("%Y-%m"),
            prev_month=f"{prev_year:04d}-{prev_month:02d}",
            next_month=f"{next_year:04d}-{next_month:02d}",
            selected_day=selected_day,
            selected_day_quizzes=selected_day_quizzes,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
        )

    @app.route("/QuizManager")
    @role_required("admin")
    def quiz_manager():
        status_filter = request.args.get("status", "all")
        search = request.args.get("q", "").strip().lower()
        message = request.args.get("message", "").strip()
        all_quizzes = get_quizzes()
        quizzes = all_quizzes if status_filter == "all" else [quiz for quiz in all_quizzes if quiz["status"] == status_filter]
        if search:
            quizzes = [
                quiz
                for quiz in quizzes
                if search in quiz["title"].lower()
                or search in quiz["subject"].lower()
                or search in quiz["quiz_code"].lower()
            ]
        return render_template(
            "quiz_manager.html",
            quizzes=quizzes,
            status_filter=status_filter,
            search=search,
            message=message,
            attempts=[attempt for quiz in all_quizzes for attempt in quiz_attempts(quiz["id"])],
        )

    @app.route("/QuizAction", methods=["POST"])
    @role_required("admin")
    def quiz_action():
        quiz_id = request.form.get("quiz_id", "").strip()
        action = request.form.get("action", "").strip()
        message = ""

        if get_quiz(quiz_id) and action == "close":
            set_quiz_status(quiz_id, "closed")
            message = "Quiz closed successfully."
        elif get_quiz(quiz_id) and action == "reopen":
            set_quiz_status(quiz_id, "draft")
            return redirect_to("create_quiz", quizId=quiz_id)
        elif get_quiz(quiz_id) and action == "delete":
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
            user = get_user("admin")
            title = request.form.get("title", "").strip()
            description = request.form.get("description", "").strip()
            subject = request.form.get("subject", "").strip()
            quiz_code = request.form.get("quiz_code", "").strip().upper() or f"QUIZ-{datetime.now().strftime('%H%M%S')}"
            scheduled_start = normalize_schedule_input(request.form.get("scheduled_start", ""))
            scheduled_end = normalize_schedule_input(request.form.get("scheduled_end", ""))
            time_limit_minutes = compute_time_limit_minutes(
                scheduled_start,
                scheduled_end,
                int(request.form.get("time_limit_minutes", "15") or 15),
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
        attempts = quiz_attempts(quiz["id"])
        flags = quiz_flags(quiz["id"])
        average = round(sum(item["percentage"] for item in attempts) / len(attempts), 1) if attempts else 0
        highest = max((item["percentage"] for item in attempts), default=0)
        return render_template(
            "quiz_results.html",
            quiz=quiz,
            attempts=attempts,
            metrics={
                "submissions": len(attempts),
                "average_score": average,
                "highest_score": highest,
                "flags_count": len(flags),
            },
            all_flags=flags,
        )

    @app.route("/ActivityMonitor")
    @role_required("admin")
    def activity_monitor():
        requested_quiz_id = request.args.get("quizId", "").strip()
        live_quiz_id = request.args.get("liveQuizId", "").strip()
        live_mode = request.args.get("live") == "1"
        severity = request.args.get("severity", "all")
        reviewed = request.args.get("reviewed", "all")
        search = request.args.get("student", "").lower().strip()
        selected_student_email = request.args.get("studentEmail", "").strip().lower()
        all_quizzes = get_quizzes()
        monitorable_quizzes = [
            quiz for quiz in all_quizzes
            if quiz.get("status") == "published" and quiz.get("monitoring_enabled")
        ]

        preferred_quiz = next(
            (quiz for quiz in monitorable_quizzes if schedule_status(quiz) == "open"),
            None,
        )
        if not preferred_quiz:
            preferred_quiz = monitorable_quizzes[0] if monitorable_quizzes else (all_quizzes[0] if all_quizzes else None)

        valid_quiz_ids = {quiz["id"] for quiz in all_quizzes}
        if requested_quiz_id and requested_quiz_id != "all" and requested_quiz_id not in valid_quiz_ids:
            requested_quiz_id = ""

        quiz_id = requested_quiz_id or (preferred_quiz["id"] if preferred_quiz else "all")
        filtered_logs = []
        for quiz in all_quizzes:
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

        student_options_map: dict[str, dict] = {}
        for flag in filtered_logs:
            student_email = str(flag.get("student_email", "")).strip().lower()
            student_name = str(flag.get("student_name", "Unknown Student")).strip() or "Unknown Student"
            if not student_email:
                student_email = f"unknown:{student_name.lower()}"
            if student_email not in student_options_map:
                student_options_map[student_email] = {
                    "student_email": str(flag.get("student_email", "")).strip(),
                    "student_name": student_name,
                }
        student_options = sorted(
            student_options_map.values(),
            key=lambda item: (item["student_name"].lower(), item["student_email"].lower()),
        )

        if selected_student_email and selected_student_email not in student_options_map:
            selected_student_email = ""

        if selected_student_email:
            filtered_logs = [
                flag for flag in filtered_logs
                if (str(flag.get("student_email", "")).strip().lower() or f"unknown:{str(flag.get('student_name', '')).strip().lower()}") == selected_student_email
            ]

        quiz_lookup = {quiz["id"]: quiz for quiz in (monitorable_quizzes or all_quizzes)}
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        student_activity_map: dict[str, dict] = {}
        for flag in filtered_logs:
            student_email = str(flag.get("student_email", "")).strip().lower()
            student_name = str(flag.get("student_name", "Unknown Student")).strip() or "Unknown Student"
            student_key = student_email or f"unknown:{student_name.lower()}"
            quiz_title = quiz_lookup.get(flag["quiz_id"], {}).get("title", flag["quiz_id"])
            event_label = str(flag.get("event_type", "")).replace("_", " ").title() or "Activity Event"
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
                entry["latest_event"] = str(flag.get("event_type", "")).replace("_", " ").title()
                entry["latest_quiz"] = quiz_title

        student_activity_rows = sorted(
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

        selected_student = next(
            (row for row in student_activity_rows if (row["student_email"].strip().lower() or f"unknown:{row['student_name'].lower()}") == selected_student_email),
            None,
        )

        if live_quiz_id and live_quiz_id not in valid_quiz_ids:
            live_quiz_id = ""

        if not live_quiz_id:
            preferred_live_quiz = get_quiz(quiz_id) if quiz_id and quiz_id != "all" else preferred_quiz
            if preferred_live_quiz:
                live_quiz_id = preferred_live_quiz["id"]
            else:
                in_progress_quiz = next(
                    (quiz for quiz in monitorable_quizzes if any(attempt["status"] == "in_progress" for attempt in quiz_attempts(quiz["id"]))),
                    None,
                )
                if in_progress_quiz:
                    live_quiz_id = in_progress_quiz["id"]

        active_quiz = get_quiz(quiz_id) if quiz_id and quiz_id != "all" else preferred_quiz
        active_students = [attempt for attempt in quiz_attempts(active_quiz["id"]) if attempt["status"] == "in_progress"] if active_quiz else []
        live_quiz = get_quiz(live_quiz_id) if live_quiz_id else None
        live_logs = quiz_flags(live_quiz["id"]) if live_quiz else []
        live_students = [attempt for attempt in quiz_attempts(live_quiz["id"]) if attempt["status"] == "in_progress"] if live_quiz else []
        latest_detection_by_attempt: dict[str, dict] = {}
        for flag in live_logs:
            attempt_id = str(flag.get("attempt_id", "")).strip()
            if not attempt_id or attempt_id in latest_detection_by_attempt:
                continue
            latest_detection_by_attempt[attempt_id] = {
                "event": flag.get("event_type", ""),
                "description": flag.get("event_description", ""),
                "flag_level": flag.get("flag_level", "low"),
                "timestamp": flag.get("timestamp", ""),
            }

        live_student_rows = []
        for attempt in live_students:
            latest = latest_detection_by_attempt.get(attempt.get("id", ""), {})
            live_student_rows.append(
                {
                    "attempt_id": attempt.get("id", ""),
                    "student_name": attempt.get("student_name", "Student"),
                    "student_email": attempt.get("student_email", ""),
                    "camera_on": False,
                    "detection_status": latest.get("event") or "normal",
                    "flag_level": latest.get("flag_level", "low"),
                    "updated_at": latest.get("timestamp") or attempt.get("started_at", ""),
                }
            )

        return render_template(
            "activity_monitor.html",
            stats=activity_stats(),
            logs=filtered_logs,
            student_activity_rows=student_activity_rows,
            student_options=student_options,
            selected_student=selected_student,
            selected_student_email=selected_student_email,
            quiz_lookup=quiz_lookup,
            quizzes=monitorable_quizzes or all_quizzes,
            selected_quiz=quiz_id,
            selected_severity=severity,
            selected_reviewed=reviewed,
            search=search,
            active_quiz=active_quiz,
            active_students=active_students,
            live_mode=live_mode and bool(live_quiz),
            live_quiz=live_quiz,
            live_logs=live_logs,
            live_students=live_students,
            live_student_rows=live_student_rows,
            live_quiz_id=live_quiz_id,
            realtime_enabled=bool(socketio),
        )

    @app.route("/StudentDashboard")
    @role_required("user")
    def student_dashboard():
        user = get_user_by_id(session.get("user_id", "")) or get_user("user")
        summary = student_dashboard_summary(user["email"])
        return render_template(
            "student_dashboard.html",
            open_quiz_list=open_quizzes(),
            attempts=student_attempts(user["email"]),
            summary=summary,
            user=user,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
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

        user = get_user_by_id(session.get("user_id", "")) or get_user("user")
        attempt = get_attempt(attempt_id) if attempt_id else None
        valid_attempt = bool(
            attempt
            and quiz_id
            and attempt.get("quiz_id") == quiz_id
            and user
            and attempt.get("student_id") == user.get("id")
            and attempt.get("status") == "in_progress"
        )

        if socketio and quiz_id and valid_attempt:
            socketio.emit(
                "detection_update",
                {
                    "quizId": quiz_id,
                    "attemptId": attempt_id,
                    "studentName": attempt.get("student_name", "Student"),
                    "studentEmail": attempt.get("student_email", ""),
                    "detection": detection_status,
                },
                to=f"monitor:{quiz_id}",
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
        user = get_user_by_id(session.get("user_id", "")) or get_user("user")
        if quiz:
            access_allowed, access_message = quiz_access_state(quiz, user["id"] if user else None)
        return render_template(
            "join_quiz.html",
            quiz=quiz,
            code=code,
            access_allowed=access_allowed,
            access_message=access_message,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
        )

    @app.route("/TakeQuiz", methods=["GET", "POST"])
    @role_required("admin", "user")
    def take_quiz():
        quiz_id = request.values.get("quizId", "").strip()
        quiz = get_quiz(quiz_id) if quiz_id else None
        if not quiz:
            return redirect_to("student_dashboard")
        access_allowed, access_message = quiz_access_state(quiz, session.get("user_id", ""))
        user = get_user_by_id(session.get("user_id", "")) or get_user("user")
        submitted = False
        attempt = None
        active_attempt_id = ""

        if request.method == "GET" and access_allowed and user:
            active_attempt_id = ensure_quiz_attempt_in_progress(quiz["id"], user["id"], False)

        if request.method == "POST" and access_allowed:
            answers = {question["id"]: request.form.get(f"question_{question['id']}", "") for question in quiz["questions"]}
            consent_given = request.form.get("consent_given") == "on" or not quiz["monitoring_enabled"]
            attempt_id = request.form.get("attempt_id", "").strip()
            if attempt_id:
                attempt = get_attempt(attempt_id)
                valid_owner = bool(attempt and attempt.get("quiz_id") == quiz["id"] and attempt.get("student_id") == user["id"])
                if not valid_owner:
                    attempt_id = ""
            if not attempt_id:
                attempt_id = ensure_quiz_attempt_in_progress(quiz["id"], user["id"], consent_given)
            attempt_id = finalize_quiz_attempt(attempt_id, answers, consent_given)
            return redirect_to("take_quiz", quizId=quiz["id"], submitted=1, attemptId=attempt_id)

        if request.args.get("submitted") == "1":
            submitted = True
            attempt = get_attempt(request.args.get("attemptId", ""))

        detection_available, detection_message = get_detection_runtime_status()

        return render_template(
            "take_quiz.html",
            quiz=quiz,
            attempt=attempt,
            submitted=submitted and access_allowed,
            access_allowed=access_allowed,
            access_message=access_message,
            format_schedule=format_schedule,
            schedule_status=schedule_status,
            realtime_enabled=bool(socketio),
            current_user_name=(user or {}).get("full_name", "Student"),
            active_attempt_id=active_attempt_id,
            detection_available=detection_available,
            detection_message=detection_message,
        )

    @app.route("/UserManagement")
    @role_required("admin")
    def user_management():
        users = get_users()
        student_count = sum(1 for user in users if user["role"] == "user")
        instructor_count = sum(1 for user in users if user["role"] == "admin")
        return render_template(
            "user_management.html",
            users=users,
            total_users=len(users),
            student_count=student_count,
            instructor_count=instructor_count,
        )

    if socketio:
        def room_name(quiz_id: str) -> str:
            return f"monitor:{quiz_id}"

        def room_participants(quiz_id: str) -> dict[str, dict]:
            return _monitor_rooms.setdefault(room_name(quiz_id), {})

        def participant_payload(participant: dict) -> dict:
            return {
                "sid": participant.get("sid", ""),
                "role": participant.get("role", ""),
                "user_id": participant.get("user_id", ""),
                "attempt_id": participant.get("attempt_id", ""),
                "display_name": participant.get("display_name", ""),
                "email": participant.get("email", ""),
                "camera_on": bool(participant.get("camera_on", False)),
            }

        def emit_monitor_student_snapshot(quiz_id: str, target_sid: str | None = None) -> None:
            participants = room_participants(quiz_id)
            student_items = []
            for item in participants.values():
                if item.get("role") != "user":
                    continue
                student_items.append(
                    {
                        "sid": item.get("sid", ""),
                        "attempt_id": item.get("attempt_id", ""),
                        "user_id": item.get("user_id", ""),
                        "display_name": item.get("display_name", "Student"),
                        "email": item.get("email", ""),
                        "camera_on": bool(item.get("camera_on", False)),
                    }
                )
            payload = {"quizId": quiz_id, "students": student_items}
            if target_sid:
                emit("monitor_students_snapshot", payload, to=target_sid)
                return
            emit("monitor_students_snapshot", payload, room=room_name(quiz_id))

        @socketio.on("join_monitor_room")
        def on_join_monitor_room(data):
            quiz_id = str((data or {}).get("quizId", "")).strip()
            if not quiz_id:
                emit("monitor_error", {"message": "Missing quiz ID."})
                return

            role = session.get("role")
            if role not in {"admin", "user"}:
                emit("monitor_error", {"message": "Unauthorized realtime connection."})
                return

            room = room_name(quiz_id)
            join_room(room)

            participants = room_participants(quiz_id)
            sid = request.sid
            current_user = get_user_by_id(session.get("user_id", "")) if session.get("user_id") else None
            display_name = ((current_user or {}).get("full_name") or role.title()).strip()
            attempt_id = str((data or {}).get("attemptId", "")).strip()
            user_id = (current_user or {}).get("id", "")
            participants[sid] = {
                "sid": sid,
                "quiz_id": quiz_id,
                "role": role,
                "user_id": user_id,
                "attempt_id": attempt_id,
                "display_name": display_name,
                "email": (current_user or {}).get("email", ""),
                "camera_on": bool((data or {}).get("cameraOn", False) and role == "user"),
            }

            emit(
                "room_snapshot",
                {
                    "quizId": quiz_id,
                    "participants": [participant_payload(item) for item in participants.values()],
                },
                to=sid,
            )
            emit_monitor_student_snapshot(quiz_id, target_sid=sid)
            emit("participant_joined", participant_payload(participants[sid]), room=room, include_self=False)
            emit_monitor_student_snapshot(quiz_id)

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
            emit_monitor_student_snapshot(quiz_id)

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

            attempt_id = str((data or {}).get("attemptId", "")).strip() or participant.get("attempt_id", "")
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
            valid_attempt = bool(attempt and attempt.get("quiz_id") == quiz_id)
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
                participants.pop(sid)
                leave_room(room)
                emit("participant_left", {"sid": sid}, room=room)
                quiz_id = room.split(":", 1)[1] if ":" in room else ""
                if quiz_id:
                    emit_monitor_student_snapshot(quiz_id)
                if not participants:
                    _monitor_rooms.pop(room, None)
                break

    return app
