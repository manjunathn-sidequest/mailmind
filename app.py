"""
AI Email Agent - FastAPI Backend
Uses Groq (free tier) with llama-3.3-70b-versatile model.
In-memory session storage only — no database.

Setup:
  pip install fastapi uvicorn groq python-dotenv google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import uuid
import base64
import logging
from email.mime.text import MIMEText
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# ── Gmail API (real integration) ─────────────────────────────────────────────
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
SCOPES        = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE    = "token.json"
CREDS_FILE    = "credentials.json"
MODEL         = "llama-3.3-70b-versatile"

# ── In-Memory Session Store ───────────────────────────────────────────────────
# sessions[session_id] = list of message dicts  (system / user / assistant / tool)
sessions: dict[str, list[dict]] = {}

SYSTEM_PROMPT = f"""You are a smart, concise personal email assistant for {GMAIL_ADDRESS or 'the user'}.
You help read, summarise, and send emails via Gmail.
When the user asks to check or read emails, ALWAYS call fetch_latest_emails.
When the user asks to send an email, ALWAYS call send_email.
Keep replies short and friendly. Never make up email content — only report what the tools return."""

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Gmail helper ──────────────────────────────────────────────────────────────
def _get_gmail_service():
    """Authenticates and returns a Gmail API service object."""
    creds = None
    
    # 1. Look for token.json on disk (local development)
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
    # 2. Render Check: If not on disk, look for the environment variable string
    elif os.getenv("GMAIL_TOKEN_CONTENT"):
        try:
            token_data = json.loads(os.getenv("GMAIL_TOKEN_CONTENT"))
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        except Exception as e:
            log.error(f"Failed to parse GMAIL_TOKEN_CONTENT environment variable: {e}")

    # 3. Handle token refresh or generation
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save the refreshed token back to disk if local
            if os.path.exists(TOKEN_FILE):
                with open(TOKEN_FILE, "w") as fh:
                    fh.write(creds.to_json())
        else:
            # Fallback wrapper to avoid freezing Render if both are missing
            if not os.path.exists(CREDS_FILE):
                raise FileNotFoundError("Both token and credentials.json are missing configuration.")
            
            # This line fails on Render, so we protect it
            log.warning("Attempting local server auth flow. This will fail on headless cloud environments.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as fh:
                fh.write(creds.to_json())
                
    return build("gmail", "v1", credentials=creds)


# ── Tool implementations ──────────────────────────────────────────────────────
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
                f"{i}. From: {headers.get('From','?')}\n"
                f"   Subject: {headers.get('Subject','(no subject)')}\n"
                f"   Date: {headers.get('Date','?')}\n"
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


# ── Groq tool definitions ─────────────────────────────────────────────────────
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
                    "recipient": {
                        "type": "string",
                        "description": "Recipient's email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Plain-text body of the email.",
                    },
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


# ── ReAct loop ────────────────────────────────────────────────────────────────
def run_react_loop(history: list[dict]) -> str:
    """
    Sends the full conversation history to Groq.
    If the model requests tool calls, executes them, appends results,
    and loops until the model returns a plain text response.
    Returns the final assistant text reply.
    Mutates `history` in-place so the caller's session is updated.
    """
    MAX_ITERATIONS = 6
    for iteration in range(MAX_ITERATIONS):
        log.info("ReAct iteration %d — sending %d messages", iteration + 1, len(history))
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1024,
        )
        choice = response.choices[0]
        message = choice.message

        # Always append the raw assistant message (may contain tool_calls)
        history.append(message.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            for tc in message.tool_calls:
                fn_name   = tc.function.name
                fn_args   = json.loads(tc.function.arguments or "{}")
                log.info("Tool call: %s(%s)", fn_name, fn_args)

                fn = TOOL_MAP.get(fn_name)
                if fn:
                    result = fn(**fn_args)
                else:
                    result = f"[Unknown tool: {fn_name}]"

                log.info("Tool result: %s", result[:120])
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         fn_name,
                    "content":      result,
                })
        else:
            # Model gave a final text response
            return message.content or "(no response)"

    return "I've been thinking for a while — could you rephrase your request?"


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Email Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message:    str


class ChatResponse(BaseModel):
    session_id: str
    reply:      str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    sid = req.session_id or str(uuid.uuid4())

    # Initialise session with system prompt on first message
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


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Debug endpoint — returns raw message history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear / reset a session."""
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "active_sessions": len(sessions)}


# Serve the frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("index.html")