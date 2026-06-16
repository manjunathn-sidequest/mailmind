"""
AI Email Agent v2 — FastAPI Backend
Normal mode  : Groq LLM + Gmail tools (unchanged from v1)
Special mode : Sarvam STT-Translate → Groq ReAct → Sarvam Translate → Sarvam TTS

Setup:
  pip install fastapi uvicorn groq python-dotenv httpx python-multipart \
              google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

Run:
  uvicorn app_2:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import uuid
import base64
import asyncio
import logging
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


# Core
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
MODEL         = "llama-3.3-70b-versatile"

# Gmail OAuth
SCOPES     = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "")
# Target Indian language for translation + TTS.

SARVAM_TARGET_LANG = os.getenv("SARVAM_TARGET_LANG", "hi-IN")
# TTS speaker voice.

SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "ritu")

# Sarvam API base URLs
_SARVAM_STT_URL       = "https://api.sarvam.ai/speech-to-text-translate"
_SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
_SARVAM_TTS_URL       = "https://api.sarvam.ai/text-to-speech"


# sessions[session_id]   = message list for normal /chat
# sessions["v_"+sid]     = message list for voice /chat-voice (separate namespace)
sessions: dict[str, list[dict]] = {}

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a smart, concise personal email assistant for {GMAIL_ADDRESS or 'the user'}.
You help read, summarise, and send emails via Gmail.
When the user asks to check or read emails, ALWAYS call fetch_latest_emails.
When the user asks to send an email, ALWAYS call send_email.
Keep replies short and friendly. Never make up email content — only report what the tools return."""

VOICE_SYSTEM_PROMPT = f"""You are a voice assistant for {GMAIL_ADDRESS or 'the user'}.
Your replies will be translated into an Indian language and spoken aloud.
STRICT OUTPUT RULES — no exceptions:
- Maximum 3 sentences OR 200 characters. Whichever is shorter.
- Zero lists, bullet points, or numbered items — summarise concisely instead.
- When asked to check emails, ALWAYS call fetch_latest_emails;
- When asked to send an email, ALWAYS call send_email;
- Never fabricate email content.
- Your reponse will be translated to indian language, so keep it easy to translate as it has to sound correctly after translation"""

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ══════════════════════════════════════════════════════════════════════════════
# ── GMAIL AUTHENTICATION & SERVICE ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _get_gmail_service():
    """Authenticate and return a Gmail API service object.
    Supports local token.json file AND a GMAIL_TOKEN_CONTENT env-var
    (for cloud deployments where the filesystem is ephemeral)."""
    creds = None

    # 1. Local token file
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # 2. Environment variable (cloud / Render)
    elif os.getenv("GMAIL_TOKEN_CONTENT"):
        try:
            token_data = json.loads(os.getenv("GMAIL_TOKEN_CONTENT"))
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        except Exception as exc:
            log.error("Failed to parse GMAIL_TOKEN_CONTENT: %s", exc)

    # 3. Refresh or run new OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if os.path.exists(TOKEN_FILE):
                with open(TOKEN_FILE, "w") as fh:
                    fh.write(creds.to_json())
        else:
            if not os.path.exists(CREDS_FILE):
                raise FileNotFoundError(
                    "credentials.json not found and no GMAIL_TOKEN_CONTENT env-var set. "
                    "See README for Gmail API setup."
                )
            log.warning("Running local OAuth flow (will fail on headless servers).")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as fh:
                fh.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ══════════════════════════════════════════════════════════════════════════════
# ── GMAIL TOOL FUNCTIONS ──────────────────────────────────────────────────────
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
    """Send an email via Gmail."""
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


# ── Groq tool schema ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_latest_emails",
            "description": "Fetch the latest emails from the user's Gmail inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent emails to fetch (1–20). Default 5.",
                        "default": 5,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email from the user's Gmail account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Recipient email address."},
                    "subject":   {"type": "string", "description": "Email subject line."},
                    "body":      {"type": "string", "description": "Plain-text body."},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
]

TOOL_MAP: dict[str, Any] = {
    "fetch_latest_emails": fetch_latest_emails,
    "send_email":          send_email,
}


# ══════════════════════════════════════════════════════════════════════════════
# ── GROQ REACT LOOP ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_react_loop(history: list[dict]) -> str:
    """
    Sends the full conversation history to Groq.
    Handles tool calls in a loop (ReAct pattern) until the model
    returns a plain-text response.
    Mutates `history` in-place — the caller's session stays up to date.
    """
    MAX_ITERATIONS = 6
    for iteration in range(MAX_ITERATIONS):
        log.info("ReAct iteration %d — %d messages in history", iteration + 1, len(history))
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

        # Append the raw assistant turn (may contain tool_calls)
        history.append(message.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                log.info("Tool call: %s(%s)", fn_name, fn_args)

                fn     = TOOL_MAP.get(fn_name)
                result = fn(**fn_args) if fn else f"[Unknown tool: {fn_name}]"

                log.info("Tool result: %s", result[:120])
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         fn_name,
                    "content":      result,
                })
        else:
            return message.content or "(no response)"

    return "I've been thinking for a while — could you rephrase your request?"


# ══════════════════════════════════════════════════════════════════════════════
# ── SARVAM AI HELPERS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _sarvam_headers_json() -> dict:
    return {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }


def sarvam_stt_to_english(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, str]:
    """
    Send raw audio bytes to Sarvam's speech-to-text-translate endpoint.
    Accepts any Indic/Hinglish speech and returns the English transcript directly.

    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    # Determine MIME type from filename extension
    ext  = filename.rsplit(".", 1)[-1].lower()
    mime = {
        "webm": "audio/webm",
        "wav":  "audio/wav",
        "mp3":  "audio/mpeg",
        "ogg":  "audio/ogg",
        "mp4":  "audio/mp4",
        "m4a":  "audio/mp4",
    }.get(ext, "audio/webm")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    files   = {"file": (filename, audio_bytes, mime)}
    data    = {"model": "saaras:v3", "with_disfluencies": "false"}

    log.info("Sarvam STT request — file: %s  mime: %s  size: %d bytes", filename, mime, len(audio_bytes))
    with httpx.Client(timeout=60) as client:
        resp = client.post(_SARVAM_STT_URL, headers=headers, files=files, data=data)

    if resp.status_code != 200:
        log.error("Sarvam STT error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result     = resp.json()
    transcript = result.get("transcript", "").strip()
    lang_code  = result.get("language_code", "hi-IN") # Extract language
    log.info("Sarvam STT transcript [%s]: %s", lang_code, transcript[:120])
    
    return transcript, lang_code # Return as a tuple


def sarvam_translate_to_indic(english_text: str, target_lang: str) -> str:
    """
    Translate an English string to the configured target Indian language
    using Sarvam's translation API.

    Sarvam docs: https://docs.sarvam.ai/api-reference-docs/endpoints/translate
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    payload = {
        "input":                english_text,
        "source_language_code": "en-IN",
        "target_language_code": target_lang, # Use dynamic language
        "speaker_gender":       "Female",
        "mode":                 "formal",
        "enable_preprocessing": True,
    }

    log.info("Sarvam Translate — target: %s  text: %s", SARVAM_TARGET_LANG, english_text[:80])
    with httpx.Client(timeout=30) as client:
        resp = client.post(_SARVAM_TRANSLATE_URL, headers=_sarvam_headers_json(), json=payload)

    if resp.status_code != 200:
        log.error("Sarvam Translate error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result       = resp.json()
    indic_text   = result.get("translated_text", english_text).strip()
    log.info("Sarvam Translate result: %s", indic_text[:120])
    return indic_text


def sarvam_tts(indic_text: str, target_lang: str) -> str:
    """
    Convert an Indic-language string to speech using Sarvam's TTS API.
    Returns a Base64-encoded WAV audio string ready for the HTML5 Audio API.

    Sarvam docs: https://docs.sarvam.ai/api-reference-docs/endpoints/text-to-speech
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    payload = {
        "inputs":               [indic_text],
        "target_language_code": target_lang, # Use dynamic language
        "speaker":              SARVAM_TTS_SPEAKER,
        "pace":                 1.0,
        "speech_sample_rate":   22050,
        "enable_preprocessing": True,
        "model":                "bulbul:v3",
    }

    log.info("Sarvam TTS — speaker: %s  lang: %s  text: %s",
             SARVAM_TTS_SPEAKER, SARVAM_TARGET_LANG, indic_text[:80])
    with httpx.Client(timeout=60) as client:
        resp = client.post(_SARVAM_TTS_URL, headers=_sarvam_headers_json(), json=payload)

    if resp.status_code != 200:
        log.error("Sarvam TTS error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    result = resp.json()
    audios = result.get("audios", [])
    if not audios:
        raise ValueError("Sarvam TTS returned no audio data.")

    # Sarvam returns the audio already base64-encoded inside the JSON
    audio_b64 = audios[0]
    log.info("Sarvam TTS audio received — base64 length: %d chars", len(audio_b64))
    return audio_b64


# ══════════════════════════════════════════════════════════════════════════════
# ── FASTAPI APPLICATION ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AI Email Agent v2", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    message:    str


class ChatResponse(BaseModel):
    session_id: str
    reply:      str


# ══════════════════════════════════════════════════════════════════════════════
# ── ENDPOINT: /chat  (Normal mode — text in, text out) ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Standard text chat using Groq + Gmail tools."""
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
# ── ENDPOINT: /chat-voice  (Special / Indic mode — audio in, audio out) ──────
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat-voice")
async def chat_voice(
    session_id: str      = Form(None),
    audio:      UploadFile = File(..., description="Recorded audio blob (webm/wav/ogg)"),
):
    """
    Full Sarvam AI pipeline:
      1. Sarvam STT-Translate  → Indic audio  →  English transcript
      2. Groq ReAct loop       → English text  →  English reply
      3. Sarvam Translate      → English reply →  Indic text
      4. Sarvam TTS            → Indic text    →  Base64 WAV audio

    Returns:
      { "session_id": str, "reply_text": str, "audio_base64": str }
    """
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is not configured. Add it to your .env file.",
        )

    # ── Read uploaded audio ───────────────────────────────────────────────────
    audio_bytes = await audio.read()
    filename    = audio.filename or "recording.webm"

    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file is too small or empty.")

    log.info("/chat-voice — file: %s  size: %d bytes  session: %s",
             filename, len(audio_bytes), session_id)

    # ── Session management (voice sessions use 'v_' prefix namespace) ─────────
    sid       = session_id or str(uuid.uuid4())
    voice_sid = f"v_{sid}"

    if voice_sid not in sessions:
        sessions[voice_sid] = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]

    history = sessions[voice_sid]

    # ── Run the full pipeline in a thread pool (all steps are blocking I/O) ───
    loop = asyncio.get_event_loop()

    try:
        # Step 1 — Sarvam STT: Indic audio → English transcript & Lang Code
        transcript, detected_lang = await loop.run_in_executor(
            None, lambda: sarvam_stt_to_english(audio_bytes, filename)
        )
        if not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not transcribe audio. Please speak more clearly and try again.",
            )
            
        # Fallback safeguard: If STT detects English ('en-IN') or an unknown code, default to Hindi
        valid_langs = ["hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN"]
        dynamic_target_lang = detected_lang if detected_lang in valid_langs else SARVAM_TARGET_LANG
        
        log.info("Step 1 STT done: '%s' (Detected: %s)", transcript[:100], dynamic_target_lang)

        # Step 2 — Groq ReAct: English transcript → English reply
        history.append({"role": "user", "content": transcript.strip()})
        english_reply = await loop.run_in_executor(
            None, lambda: run_react_loop(history)
        )
        log.info("Step 2 Groq done: '%s'", english_reply[:100])

        # Step 3 — Sarvam Translate: English reply → Dynamic Indic text
        indic_text = await loop.run_in_executor(
            None, lambda: sarvam_translate_to_indic(english_reply, dynamic_target_lang)
        )
        log.info("Step 3 Translate done: '%s'", indic_text[:100])

        # Step 4 — Sarvam TTS: Indic text → Base64 WAV audio
        audio_b64 = await loop.run_in_executor(
            None, lambda: sarvam_tts(indic_text, dynamic_target_lang)
        )
        log.info("Step 4 TTS done — base64 length: %d", len(audio_b64))

        return {
            "session_id":   sid,
            "reply_text":   indic_text,
            "audio_base64": audio_b64,
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
# ── UTILITY ENDPOINTS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Debug — return raw message history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Reset a session (both text and voice sub-sessions)."""
    sessions.pop(session_id, None)
    sessions.pop(f"v_{session_id}", None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model":           MODEL,
        "active_sessions": len(sessions),
        "sarvam_key_set":  bool(SARVAM_API_KEY),
        "sarvam_lang":     SARVAM_TARGET_LANG,
        "sarvam_speaker":  SARVAM_TTS_SPEAKER,
    }


# ── Serve the v2 frontend ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("index.html")
