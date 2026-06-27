"""
AI Email + Voice Calling Agent — FastAPI Backend  (v3)
────────────────────────────────────────────────────────
Normal mode  : Groq LLM + Gmail tools
Special mode : Sarvam STT-Translate → Groq ReAct → Sarvam Translate → Sarvam TTS
Calling mode : Groq tool → SQLite contacts lookup → LiveKit/Twilio outbound call

New in v3
─────────
  • SQLite `contacts.db`  — persistent contact directory (name + phone)
  • Groq tool: save_contact        — AI saves contacts mid-conversation
  • Groq tool: initiate_outbound_call — AI dispatches calls via Calling Agent
  • REST  POST /contacts            — manual contact upsert
  • REST  GET  /contacts            — list all contacts
  • REST  POST /webhook/call-summary— receives post-call summary → self-email

Setup:
  pip install fastapi uvicorn groq python-dotenv httpx python-multipart \
              google-auth google-auth-oauthlib google-auth-httplib2 \
              google-api-python-client

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Required .env keys:
  GROQ_API_KEY, GMAIL_ADDRESS,
  SARVAM_API_KEY, SARVAM_TARGET_LANG, SARVAM_TTS_SPEAKER,
  CALLING_AGENT_URL   (default: http://localhost:8001)
"""

import os
import json
import uuid
import base64
import sqlite3
import asyncio
import logging
from contextlib import asynccontextmanager
from email.mime.text import MIMEText
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# ── Gmail API ─────────────────────────────────────────────────────────────────
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Core LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
MODEL = "llama-3.3-70b-versatile"

# Gmail OAuth
SCOPES     = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

# Sarvam AI (Indic voice mode)
SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "")
SARVAM_TARGET_LANG = os.getenv("SARVAM_TARGET_LANG", "hi-IN")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "ritu")

# Sarvam API endpoints
_SARVAM_STT_URL       = "https://api.sarvam.ai/speech-to-text-translate"
_SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
_SARVAM_TTS_URL       = "https://api.sarvam.ai/text-to-speech"

# Calling Agent microservice URL (calling_agent.py runs separately on port 8001)
CALLING_AGENT_URL = os.getenv("CALLING_AGENT_URL", "http://localhost:8001")

# SQLite database path
DB_PATH = "contacts.db"


# ══════════════════════════════════════════════════════════════════════════════
# ── SQLITE CONTACTS DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """
    Create the contacts table on first run.
    COLLATE NOCASE on the name column ensures case-insensitive UNIQUE
    constraint so 'John' and 'john' are treated as the same contact.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT    NOT NULL COLLATE NOCASE,
                phone_number TEXT    NOT NULL,
                UNIQUE(name COLLATE NOCASE)
            )
        """)
        conn.commit()
    log.info("Contacts DB ready at '%s'", DB_PATH)


def _get_db() -> sqlite3.Connection:
    """Return a connection with dict-like Row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── FastAPI lifespan: init DB before accepting requests ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# ══════════════════════════════════════════════════════════════════════════════
# ── IN-MEMORY SESSION STORE
# ══════════════════════════════════════════════════════════════════════════════
# sessions[session_id]    = message list for /chat (text mode)
# sessions["v_"+sid]      = message list for /chat-voice (Indic voice mode)
sessions: dict[str, list[dict]] = {}


# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a smart, concise personal assistant for {GMAIL_ADDRESS or 'the user'}.
You can read and send emails via Gmail, save contacts to a directory, and place outbound AI voice calls.

TOOL USAGE RULES:
- To check/read emails          → ALWAYS call fetch_latest_emails
- To send an email              → ALWAYS call send_email
- When user gives a name + phone number to save → ALWAYS call save_contact
- To call someone               → ALWAYS call initiate_outbound_call
  (If the contact is not in the directory, ask the user for their phone number first, save it, then call.)

Keep replies short and friendly. Never fabricate data."""

VOICE_SYSTEM_PROMPT = f"""You are a voice assistant for {GMAIL_ADDRESS or 'the user'}.
Your replies will be translated into an Indian language and spoken aloud.
STRICT OUTPUT RULES — no exceptions:
- Maximum 3 sentences OR 200 characters. Whichever is shorter.
- Zero lists, bullet points, or numbered items — summarise concisely instead.
- When asked to check emails, ALWAYS call fetch_latest_emails.
- When asked to send an email, ALWAYS call send_email.
- When asked to call someone, ALWAYS call initiate_outbound_call.
- Never fabricate content.
- Keep language simple so it translates cleanly into Indian languages."""


# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# ── GMAIL AUTHENTICATION & SERVICE
# ══════════════════════════════════════════════════════════════════════════════

def _get_gmail_service():
    """
    Return an authenticated Gmail API service object.
    Strategy:
      1. Local token.json file (development)
      2. GMAIL_TOKEN_CONTENT env-var (cloud/Render where filesystem is ephemeral)
      3. Run OAuth flow (first-time local setup only)
    """
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    elif os.getenv("GMAIL_TOKEN_CONTENT"):
        try:
            token_data = json.loads(os.getenv("GMAIL_TOKEN_CONTENT"))
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        except Exception as exc:
            log.error("Failed to parse GMAIL_TOKEN_CONTENT: %s", exc)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if os.path.exists(TOKEN_FILE):
                with open(TOKEN_FILE, "w") as fh:
                    fh.write(creds.to_json())
        else:
            if not os.path.exists(CREDS_FILE):
                raise FileNotFoundError(
                    "credentials.json not found and GMAIL_TOKEN_CONTENT is not set. "
                    "See README for Gmail API setup instructions."
                )
            log.warning("Running local OAuth flow — this will fail on headless servers.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as fh:
                fh.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ══════════════════════════════════════════════════════════════════════════════
# ── TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_latest_emails(count: int = 5) -> str:
    """Fetch the latest `count` emails from Gmail inbox."""
    count = max(1, min(count, 20))
    try:
        service = _get_gmail_service()
        results = service.users().messages().list(
            userId="me", labelIds=["INBOX"], maxResults=count
        ).execute()
        messages = results.get("messages", [])
        if not messages:
            return "Your inbox is empty."

        output_lines = []
        for i, msg_ref in enumerate(messages, 1):
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            snippet = msg.get("snippet", "")[:120]
            output_lines.append(
                f"{i}. From: {headers.get('From', '?')}\n"
                f"   Subject: {headers.get('Subject', '(no subject)')}\n"
                f"   Date: {headers.get('Date', '?')}\n"
                f"   Preview: {snippet}…"
            )
        return "\n\n".join(output_lines)

    except FileNotFoundError as exc:
        return f"[Gmail not connected] {exc}"
    except Exception as exc:
        log.exception("fetch_latest_emails error")
        return f"[Error fetching emails] {exc}"


def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email via the authenticated Gmail account."""
    try:
        service = _get_gmail_service()
        mime = MIMEText(body)
        mime["to"]      = recipient
        mime["from"]    = GMAIL_ADDRESS
        mime["subject"] = subject
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        return f"Email sent successfully to {recipient} with subject '{subject}'."
    except FileNotFoundError as exc:
        return f"[Gmail not connected] {exc}"
    except Exception as exc:
        log.exception("send_email error")
        return f"[Error sending email] {exc}"


def save_contact(name: str, phone_number: str) -> str:
    """
    Save a new contact or update an existing one in the local SQLite directory.
    Uses an UPSERT so calling with an existing name just updates the number.
    """
    name         = name.strip()
    phone_number = phone_number.strip()

    if not name or not phone_number:
        return "[Error] Both name and phone_number are required."

    try:
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO contacts (name, phone_number)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET phone_number = excluded.phone_number
                """,
                (name, phone_number),
            )
            conn.commit()
        log.info("Contact saved: %s → %s", name, phone_number)
        return f"Contact '{name}' saved with phone number {phone_number}."
    except Exception as exc:
        log.exception("save_contact error")
        return f"[Error saving contact] {exc}"


def initiate_outbound_call(contact_name: str, initial_message: str) -> str:
    """
    Look up `contact_name` in the SQLite directory (case-insensitive),
    then dispatch an outbound AI voice call via the Calling Agent microservice.

    Returns a human-readable status string for Groq to relay to the user.
    """
    contact_name    = contact_name.strip()
    initial_message = initial_message.strip()

    # ── Step 1: Lookup contact in the local database ──────────────────────────
    try:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT name, phone_number FROM contacts WHERE name = ? COLLATE NOCASE",
                (contact_name,),
            ).fetchone()
    except Exception as exc:
        log.exception("DB lookup error in initiate_outbound_call")
        return f"[Database Error] Could not search contacts: {exc}"

    if not row:
        return (
            f"Contact '{contact_name}' not found in the directory. "
            f"Please ask the user for {contact_name}'s phone number, "
            f"save it with save_contact, and then try calling again."
        )

    phone_number  = row["phone_number"]
    resolved_name = row["name"]          # use the stored casing

    # ── Step 2: POST to the Calling Agent microservice ────────────────────────
    payload = {
        "phone_number":    phone_number,
        "contact_name":    resolved_name,
        "initial_message": initial_message,
    }

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(f"{CALLING_AGENT_URL}/trigger-call", json=payload)
            resp.raise_for_status()

        log.info("Call dispatched for %s (%s)", resolved_name, phone_number)
        return (
            f"Outbound call successfully queued for {resolved_name} ({phone_number}). "
            f"The AI Voice Agent is dialling now."
        )

    except httpx.ConnectError:
        return (
            f"[Calling Agent Offline] Could not reach the calling service at {CALLING_AGENT_URL}. "
            f"Make sure calling_agent.py is running on port 8001."
        )
    except httpx.HTTPStatusError as exc:
        return f"[Calling Agent Error {exc.response.status_code}] {exc.response.text[:200]}"
    except Exception as exc:
        log.exception("initiate_outbound_call HTTP error")
        return f"[Call Error] {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# ── GROQ TOOL SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict] = [
    # ── Email: read ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name":        "fetch_latest_emails",
            "description": "Fetch the latest emails from the user's Gmail inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type":        "integer",
                        "description": "Number of recent emails to fetch (1–20). Default 5.",
                        "default":     5,
                    }
                },
                "required": [],
            },
        },
    },
    # ── Email: send ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name":        "send_email",
            "description": "Send an email from the user's Gmail account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Recipient's email address."},
                    "subject":   {"type": "string", "description": "Email subject line."},
                    "body":      {"type": "string", "description": "Plain-text body of the email."},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
    # ── Contacts: save ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_contact",
            "description": (
                "Save a new contact or update an existing one in the personal directory. "
                "Call this whenever the user provides a name and phone number to store."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type":        "string",
                        "description": "Full name of the contact (e.g. 'John Doe').",
                    },
                    "phone_number": {
                        "type":        "string",
                        "description": (
                            "Phone number with full country code. "
                            "Examples: +919876543210 (India), +14155552671 (US)."
                        ),
                    },
                },
                "required": ["name", "phone_number"],
            },
        },
    },
    # ── Calling: dispatch ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "initiate_outbound_call",
            "description": (
                "Place an AI-powered outbound voice call to a saved contact. "
                "The AI voice agent will deliver `initial_message` in spoken form. "
                "Use this whenever the user says 'call', 'phone', or 'ring' someone. "
                "If the contact is not yet saved, ask for their number, save it first, then call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type":        "string",
                        "description": "Name of the contact to call. Must already exist in the directory.",
                    },
                    "initial_message": {
                        "type":        "string",
                        "description": "The message the AI voice agent will speak to the contact.",
                    },
                },
                "required": ["contact_name", "initial_message"],
            },
        },
    },
]

# Maps tool name → Python function
TOOL_MAP: dict[str, Any] = {
    "fetch_latest_emails":   fetch_latest_emails,
    "send_email":            send_email,
    "save_contact":          save_contact,
    "initiate_outbound_call": initiate_outbound_call,
}


# ══════════════════════════════════════════════════════════════════════════════
# ── GROQ REACT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_react_loop(history: list[dict]) -> str:
    """
    Send the full conversation history to Groq and handle tool calls in a loop
    (ReAct pattern) until the model returns a plain-text reply.
    Mutates `history` in-place so the caller's session stays current.
    """
    MAX_ITERATIONS = 8   # raised to 8 to handle save-then-call two-step flows
    for iteration in range(MAX_ITERATIONS):
        log.info("ReAct iteration %d — %d messages", iteration + 1, len(history))
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1024,
        )
        choice  = response.choices[0]
        message = choice.message

        # Always append the raw assistant message (may carry tool_calls)
        history.append(message.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                log.info("Tool call: %s(%s)", fn_name, fn_args)

                fn     = TOOL_MAP.get(fn_name)
                result = fn(**fn_args) if fn else f"[Unknown tool: {fn_name}]"

                log.info("Tool result: %s", result[:150])
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         fn_name,
                    "content":      result,
                })
        else:
            return message.content or "(no response)"

    return "I have been thinking for a while — could you rephrase your request?"


# ══════════════════════════════════════════════════════════════════════════════
# ── SARVAM AI HELPERS  (Indic voice mode)
# ══════════════════════════════════════════════════════════════════════════════

def _sarvam_headers_json() -> dict:
    return {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }


def sarvam_stt_to_english(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, str]:
    """
    Transcribe Indic/Hinglish audio and translate it to English in one API call.
    The model (saaras:v3) auto-detects the spoken language.

    Returns:
        (english_transcript, detected_language_code)
        e.g. ("Check my emails", "kn-IN")

    Supported input languages (auto-detected):
        hi-IN Hindi | kn-IN Kannada | ta-IN Tamil | te-IN Telugu
        ml-IN Malayalam | mr-IN Marathi | bn-IN Bengali
        gu-IN Gujarati | pa-IN Punjabi | od-IN Odia
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    ext  = filename.rsplit(".", 1)[-1].lower()
    mime = {
        "webm": "audio/webm", "wav": "audio/wav",
        "mp3":  "audio/mpeg", "ogg": "audio/ogg",
        "mp4":  "audio/mp4",  "m4a": "audio/mp4",
    }.get(ext, "audio/webm")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    files   = {"file": (filename, audio_bytes, mime)}
    data    = {"model": "saaras:v3", "with_disfluencies": "false"}

    log.info("Sarvam STT — file: %s  mime: %s  size: %d bytes", filename, mime, len(audio_bytes))
    with httpx.Client(timeout=60) as client:
        resp = client.post(_SARVAM_STT_URL, headers=headers, files=files, data=data)

    if resp.status_code != 200:
        log.error("Sarvam STT error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result     = resp.json()
    transcript = result.get("transcript", "").strip()
    lang_code  = result.get("language_code", "hi-IN").strip() or SARVAM_TARGET_LANG
    log.info("Sarvam STT transcript [detected: %s]: %s", lang_code, transcript[:120])
    return transcript, lang_code


def sarvam_translate_to_indic(english_text: str, target_lang: str) -> str:
    """Translate English text to `target_lang` using Sarvam's translate API."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    payload = {
        "input":                english_text,
        "source_language_code": "en-IN",
        "target_language_code": target_lang,
        "speaker_gender":       "Female",
        "mode":                 "formal",
        "enable_preprocessing": True,
    }
    log.info("Sarvam Translate — target: %s  text: %s", target_lang, english_text[:80])
    with httpx.Client(timeout=30) as client:
        resp = client.post(_SARVAM_TRANSLATE_URL, headers=_sarvam_headers_json(), json=payload)

    if resp.status_code != 200:
        log.error("Sarvam Translate error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result = resp.json()
    indic  = result.get("translated_text", english_text).strip()
    log.info("Sarvam Translate result [%s]: %s", target_lang, indic[:120])
    return indic


def sarvam_tts(indic_text: str, target_lang: str) -> str:
    """
    Convert Indic text to speech. Returns a Base64-encoded WAV string
    ready for the browser's HTML5 Audio API.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    payload = {
        "inputs":               [indic_text],
        "target_language_code": target_lang,
        "speaker":              SARVAM_TTS_SPEAKER,
        "pace":                 1.0,
        "speech_sample_rate":   22050,
        "enable_preprocessing": True,
        "model":                "bulbul:v3",
    }
    log.info("Sarvam TTS — speaker: %s  lang: %s  text: %s",
             SARVAM_TTS_SPEAKER, target_lang, indic_text[:80])
    with httpx.Client(timeout=60) as client:
        resp = client.post(_SARVAM_TTS_URL, headers=_sarvam_headers_json(), json=payload)

    if resp.status_code != 200:
        log.error("Sarvam TTS error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result = resp.json()
    audios = result.get("audios", [])
    if not audios:
        raise ValueError("Sarvam TTS returned no audio data.")

    audio_b64 = audios[0]
    log.info("Sarvam TTS audio received [%s] — b64 length: %d", target_lang, len(audio_b64))
    return audio_b64


# ══════════════════════════════════════════════════════════════════════════════
# ── FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Email + Calling Agent",
    version="3.0",
    lifespan=lifespan,          # initialises SQLite on startup
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    message:    str

class ChatResponse(BaseModel):
    session_id: str
    reply:      str

class ContactRequest(BaseModel):
    name:         str
    phone_number: str

class CallSummaryWebhook(BaseModel):
    contact_name: str
    summary:      str


# ══════════════════════════════════════════════════════════════════════════════
# ── ENDPOINT: /contacts  (Contact directory REST API)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/contacts", status_code=201)
def add_contact_rest(req: ContactRequest):
    """
    Manually upsert a contact via REST (useful for testing or bulk import).
    The AI can also save contacts automatically via the save_contact tool.
    """
    if not req.name.strip() or not req.phone_number.strip():
        raise HTTPException(status_code=400, detail="Both 'name' and 'phone_number' are required.")
    result = save_contact(req.name, req.phone_number)
    if result.startswith("[Error"):
        raise HTTPException(status_code=500, detail=result)
    return {"status": "ok", "detail": result}


@app.get("/contacts")
def list_contacts():
    """Return all saved contacts ordered alphabetically by name."""
    try:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT id, name, phone_number FROM contacts ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return {"contacts": [dict(r) for r in rows]}
    except Exception as exc:
        log.exception("list_contacts error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/contacts/{name}")
def delete_contact(name: str):
    """Remove a contact by name (case-insensitive)."""
    try:
        with _get_db() as conn:
            cur = conn.execute(
                "DELETE FROM contacts WHERE name = ? COLLATE NOCASE", (name,)
            )
            conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Contact '{name}' not found.")
        return {"status": "deleted", "name": name}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# ── ENDPOINT: /chat  (Normal mode — text in, text out)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Standard text chat using Groq + Gmail + Contacts + Calling tools."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    sid = req.session_id or str(uuid.uuid4())

    if sid not in sessions:
        sessions[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = sessions[sid]
    history.append({"role": "user", "content": req.message.strip()})

    try:
        reply = run_react_loop(history)
    except Exception as exc:
        log.exception("Error in ReAct loop")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(session_id=sid, reply=reply)


# ══════════════════════════════════════════════════════════════════════════════
# ── ENDPOINT: /chat-voice  (Indic mode — audio in, audio out via Sarvam)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat-voice")
async def chat_voice(
    session_id: str        = Form(None),
    audio:      UploadFile = File(..., description="Recorded audio blob (webm/wav/ogg)"),
):
    """
    Full Sarvam AI pipeline:
      1. Sarvam STT-Translate  → Indic audio       → English transcript + detected lang
      2. Groq ReAct loop       → English transcript → English reply (with all tools)
      3. Sarvam Translate      → English reply      → Indic text
      4. Sarvam TTS            → Indic text         → Base64 WAV audio

    Returns: { "session_id", "reply_text", "audio_base64", "detected_language" }
    """
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is not configured. Add it to your .env file.",
        )

    audio_bytes = await audio.read()
    filename    = audio.filename or "recording.webm"

    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file is too small or empty.")

    log.info("/chat-voice — file: %s  size: %d bytes  session: %s",
             filename, len(audio_bytes), session_id)

    sid       = session_id or str(uuid.uuid4())
    voice_sid = f"v_{sid}"

    if voice_sid not in sessions:
        sessions[voice_sid] = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]

    history = sessions[voice_sid]
    loop    = asyncio.get_event_loop()

    # Fallback-safe list of Indic language codes Sarvam supports for output
    VALID_OUTPUT_LANGS = {
        "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
        "mr-IN", "bn-IN", "gu-IN", "pa-IN", "od-IN",
    }

    try:
        # Step 1 — Sarvam STT: audio → English + detected language
        transcript, detected_lang = await loop.run_in_executor(
            None, lambda: sarvam_stt_to_english(audio_bytes, filename)
        )
        if not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not transcribe audio. Please speak more clearly.",
            )

        # Use detected language for response; fall back to configured default
        target_lang = detected_lang if detected_lang in VALID_OUTPUT_LANGS else SARVAM_TARGET_LANG
        log.info("Step 1 STT done: '%s' (lang: %s → output: %s)",
                 transcript[:80], detected_lang, target_lang)

        # Step 2 — Groq ReAct: English transcript → English reply
        history.append({"role": "user", "content": transcript.strip()})
        english_reply = await loop.run_in_executor(
            None, lambda: run_react_loop(history)
        )
        log.info("Step 2 Groq done: '%s'", english_reply[:100])

        # Step 3 — Sarvam Translate: English → detected Indic language
        indic_text = await loop.run_in_executor(
            None, lambda: sarvam_translate_to_indic(english_reply, target_lang)
        )
        log.info("Step 3 Translate done [%s]: '%s'", target_lang, indic_text[:100])

        # Step 4 — Sarvam TTS: Indic text → Base64 WAV
        audio_b64 = await loop.run_in_executor(
            None, lambda: sarvam_tts(indic_text, target_lang)
        )
        log.info("Step 4 TTS done — b64 length: %d", len(audio_b64))

        return {
            "session_id":        sid,
            "reply_text":        indic_text,
            "audio_base64":      audio_b64,
            "detected_language": target_lang,
        }

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        log.exception("Sarvam API HTTP error")
        raise HTTPException(
            status_code=502,
            detail=f"Sarvam API error {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        log.exception("Unexpected error in /chat-voice")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# ── ENDPOINT: /webhook/call-summary  (Calling Agent → self-email)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/call-summary")
def handle_call_summary(webhook: CallSummaryWebhook):
    """
    Called by the Calling Agent microservice after a call ends.
    Receives the AI-generated call transcript summary and emails it
    to GMAIL_ADDRESS as a post-call record.
    """
    subject = f"Call Summary: AI Agent conversation with {webhook.contact_name}"
    body = (
        f"Your AI Voice Agent completed a call with {webhook.contact_name}.\n\n"
        f"--- Conversation Summary ---\n"
        f"{webhook.summary}\n\n"
        f"— Sent automatically by Mail Mind"
    )
    result = send_email(
        recipient=GMAIL_ADDRESS,
        subject=subject,
        body=body,
    )
    log.info("Call summary webhook handled. Email result: %s", result)
    return {"status": "success", "detail": "Summary email dispatched."}


# ══════════════════════════════════════════════════════════════════════════════
# ── UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Debug — return raw message history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Reset both the text and voice sub-sessions for a given ID."""
    sessions.pop(session_id, None)
    sessions.pop(f"v_{session_id}", None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
def health():
    """Health check + configuration summary."""
    try:
        with _get_db() as conn:
            contact_count = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
    except Exception:
        contact_count = -1

    return {
        "status":          "ok",
        "model":           MODEL,
        "active_sessions": len(sessions),
        "contact_count":   contact_count,
        "calling_agent":   CALLING_AGENT_URL,
        "sarvam_key_set":  bool(SARVAM_API_KEY),
        "sarvam_lang":     SARVAM_TARGET_LANG,
        "sarvam_speaker":  SARVAM_TTS_SPEAKER,
    }


# ── Serve the frontend ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("index.html")
