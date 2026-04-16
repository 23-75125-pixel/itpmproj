from __future__ import annotations

from pathlib import Path

import base64
import json
import os
import re
import ssl
import smtplib
from email.message import EmailMessage
from io import BytesIO
from datetime import datetime, timedelta
from functools import wraps
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
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
    create_or_update_quiz,
    create_user,
    dashboard_stats,
    delete_user_by_email,
    format_schedule,
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
    schedule_status,
    student_dashboard_summary,
    student_attempts,
    submit_quiz_attempt,
    verify_password,
    set_quiz_status,
)


DETECTION_MODEL_PATH = Path(__file__).resolve().parent.parent / "best.pt"
MIN_CONFIDENCE_NORMAL = 0.45
MIN_CONFIDENCE_DETECTION = 0.20
MIN_FACE_AREA_RATIO = 0.05
MAX_FACE_AREA_RATIO = 0.60
CENTER_MARGIN_RATIO = 0.25

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading") if SocketIO else None
_monitor_rooms: dict[str, dict[str, dict]] = {}


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
    init_database()

    if socketio:
        socketio.init_app(app, manage_session=False)

    detection_model = None

    def get_detection_model():
        nonlocal detection_model
        if detection_model is not None:
            return detection_model
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Install the required dependencies and restart the server.")
        if not DETECTION_MODEL_PATH.exists():
            raise RuntimeError(f"YOLO model not found at {DETECTION_MODEL_PATH}")
        detection_model = YOLO(str(DETECTION_MODEL_PATH))
        return detection_model

    def _is_google_oauth_configured() -> bool:
        return bool(os.environ.get("GOOGLE_CLIENT_ID", "").strip() and os.environ.get("GOOGLE_CLIENT_SECRET", "").strip())

    def _get_google_oauth_config():
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise RuntimeError("Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", url_for("google_callback", _external=True))
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
                    return redirect(url_for("login"))
                if role not in allowed_roles:
                    return redirect(url_for("home"))
                return view(*args, **kwargs)

            return wrapped

        return decorator

    @app.route("/")
    def index():
        if session.get("role") in {"admin", "user"}:
            return redirect(url_for("home"))
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
                return redirect(url_for("home"))

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
            return redirect(url_for("login", role=role, error="Google OAuth is not configured."))
        try:
            return redirect(_authorize_google(role))
        except RuntimeError as exc:
            return redirect(url_for("login", role=role, error=str(exc)))

    @app.route("/google-callback")
    def google_callback():
        code = request.args.get("code", "").strip()
        role = request.args.get("state", "user").strip()
        if role not in {"admin", "user"}:
            role = "user"
        if not code:
            return redirect(url_for("login", role=role, error="Google login failed."))
        try:
            profile = _fetch_google_user_info(code)
        except Exception as exc:
            return redirect(url_for("login", role=role, error=str(exc)))

        email = (profile.get("email") or "").strip().lower()
        full_name = (profile.get("name") or profile.get("given_name") or email).strip()
        if not email:
            return redirect(url_for("login", role=role, error="Google did not return a valid email address."))

        user = get_user_by_email(email)
        if user and user["role"] != role:
            return redirect(url_for("login", role=role, error="Your Google account does not match the selected portal."))

        if not user:
            if role == "admin":
                return redirect(
                    url_for(
                        "login",
                        role=role,
                        error="This instructor Google account is not authorized. Only registered instructor accounts can sign in.",
                    )
                )
            return redirect(
                url_for(
                    "login",
                    role=role,
                    error="This student Google account is not invited yet. Ask your instructor to send an invitation first.",
                )
            )

        session["role"] = role
        session["user_id"] = user["id"]
        return redirect(url_for("home"))

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
        if request.method == "POST":
            email = request.form.get("email", "").strip()
            user = get_user_by_email(email)
            if user:
                message = f"Password reset instructions were sent to {user['email']}."
            else:
                message = "If the email exists in the system, password reset instructions were sent."
        return render_template("forgot_password.html", message=message)

    @app.route("/signin/<role>")
    def signin(role: str):
        if role not in {"admin", "user"}:
            return redirect(url_for("login"))
        user = get_user(role)
        session["role"] = role
        session["user_id"] = user["id"]
        return redirect(url_for("home"))

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/home")
    def home():
        role = session.get("role")
        if role == "admin":
            return redirect(url_for("dashboard"))
        if role == "user":
            return redirect(url_for("student_dashboard"))
        return redirect(url_for("login"))

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
            set_quiz_status(quiz_id, "published")
            message = "Quiz reopened successfully."
        elif get_quiz(quiz_id) and action == "delete":
            delete_quiz_by_id(quiz_id)
            message = "Quiz deleted successfully."
        else:
            message = "Action could not be completed."

        return redirect(url_for("quiz_manager", message=message))

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
            time_limit_minutes = int(request.form.get("time_limit_minutes", "15") or 15)
            quiz_code = request.form.get("quiz_code", "").strip().upper() or f"QUIZ-{datetime.now().strftime('%H%M%S')}"
            scheduled_start = normalize_schedule_input(request.form.get("scheduled_start", ""))
            scheduled_end = normalize_schedule_input(request.form.get("scheduled_end", ""))
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
            return redirect(url_for("quiz_manager", message=message))

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
            return redirect(url_for("quiz_manager", message="Create a quiz first before viewing results."))
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
        quiz_id = request.args.get("quizId", "all")
        live_quiz_id = request.args.get("liveQuizId", "").strip()
        live_mode = request.args.get("live") == "1"
        severity = request.args.get("severity", "all")
        reviewed = request.args.get("reviewed", "all")
        search = request.args.get("student", "").lower().strip()
        all_quizzes = get_quizzes()
        filtered_logs = []
        for quiz in all_quizzes:
            filtered_logs.extend(quiz_flags(quiz["id"]))
        if quiz_id != "all":
            filtered_logs = [flag for flag in filtered_logs if flag["quiz_id"] == quiz_id]
        if severity != "all":
            filtered_logs = [flag for flag in filtered_logs if flag["flag_level"] == severity]
        if reviewed != "all":
            expected = reviewed == "reviewed"
            filtered_logs = [flag for flag in filtered_logs if flag["reviewed"] == expected]
        if search:
            filtered_logs = [flag for flag in filtered_logs if search in flag["student_name"].lower()]

        active_quiz = get_quiz(quiz_id) if quiz_id != "all" else (all_quizzes[0] if all_quizzes else None)
        active_students = [attempt for attempt in quiz_attempts(active_quiz["id"]) if attempt["status"] == "in_progress"] if active_quiz else []
        live_quiz = get_quiz(live_quiz_id) if live_quiz_id else None
        live_logs = quiz_flags(live_quiz["id"]) if live_quiz else []
        live_students = [attempt for attempt in quiz_attempts(live_quiz["id"]) if attempt["status"] == "in_progress"] if live_quiz else []
        return render_template(
            "activity_monitor.html",
            stats=activity_stats(),
            logs=filtered_logs,
            quizzes=all_quizzes,
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
        return render_template("student_camera.html")

    @app.route("/detect-face", methods=["POST"])
    @role_required("user")
    def detect_face():
        if cv2 is None or YOLO is None:
            return jsonify({"error": "Detection dependencies are not installed."}), 500

        data = request.get_json(silent=True) or {}
        image_data = str(data.get("image", ""))
        frame = decode_image_from_data_url(image_data)
        if frame is None:
            return jsonify({"error": "Unable to decode the camera frame."}), 400

        try:
            model = get_detection_model()
            results = model(frame)[0]
            frame_height, frame_width = frame.shape[:2]
            detections = []
            
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())
                label = str(model.names.get(class_id, class_id))
                
                x1, y1, x2, y2 = coords
                is_normal, cheating_reason = classify_face_detection(
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    frame_width,
                    frame_height,
                )

                detections.append(
                    {
                        "bbox": [coords[0], coords[1], coords[2], coords[3]],
                        "confidence": confidence,
                        "label": label,
                        "type": "normal" if is_normal else "cheating",
                        "cheating_reason": cheating_reason if not is_normal else None,
                    }
                )
            
            # Multiple faces detected = cheating
            if len(detections) > 1:
                for det in detections:
                    det["type"] = "cheating"
                    det["cheating_reason"] = "multiple_faces"
                    
        except Exception as exc:
            return jsonify({"error": f"Detection failed: {exc}"}), 500

        return jsonify({"detections": detections})

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
            return redirect(url_for("student_dashboard"))
        access_allowed, access_message = quiz_access_state(quiz, session.get("user_id", ""))
        user = get_user_by_id(session.get("user_id", "")) or get_user("user")
        submitted = False
        attempt = None

        if request.method == "POST" and access_allowed:
            answers = {question["id"]: request.form.get(f"question_{question['id']}", "") for question in quiz["questions"]}
            consent_given = request.form.get("consent_given") == "on" or not quiz["monitoring_enabled"]
            attempt_id = submit_quiz_attempt(quiz["id"], user["id"], answers, consent_given)
            return redirect(url_for("take_quiz", quizId=quiz["id"], submitted=1, attemptId=attempt_id))

        if request.args.get("submitted") == "1":
            submitted = True
            attempt = get_attempt(request.args.get("attemptId", ""))

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
                "display_name": participant.get("display_name", ""),
                "camera_on": bool(participant.get("camera_on", False)),
            }

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
            display_name = (session.get("full_name") or role.title()).strip()
            participants[sid] = {
                "sid": sid,
                "quiz_id": quiz_id,
                "role": role,
                "display_name": display_name,
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

            if camera_on:
                active_students = sum(
                    1
                    for item in participants.values()
                    if item.get("role") == "user" and bool(item.get("camera_on", False))
                )
                if not participant.get("camera_on") and active_students >= 10:
                    emit("monitor_error", {"message": "Camera limit reached (max 10 students)."}, to=request.sid)
                    return

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

        @socketio.on("disconnect")
        def on_disconnect():
            sid = request.sid
            for room, participants in list(_monitor_rooms.items()):
                if sid not in participants:
                    continue
                participants.pop(sid)
                leave_room(room)
                emit("participant_left", {"sid": sid}, room=room)
                if not participants:
                    _monitor_rooms.pop(room, None)
                break

    return app
