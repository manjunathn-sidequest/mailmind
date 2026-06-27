import os
import logging
import threading
import uvicorn
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit import api
from livekit.agents import WorkerAgent, JobContext, WorkerOptions, cli
from livekit.plugins import openai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("calling_agent")

app = FastAPI(title="Manjunath's Voice Outbound Microservice")

# ── GLOBAL VOICE CALL METADATA TRACKER ────────────────────────────────────────
pending_calls = {}

class OutboundCallRequest(BaseModel):
    phone_number: str
    contact_name: str
    initial_message: str

# ── FASTAPI TRIGGER ENDPOINT ──────────────────────────────────────────────────
@app.post("/trigger-call")
async def trigger_call(payload: OutboundCallRequest):
    """Invoked by app.py to dial the phone via Twilio."""
    lk_url = os.getenv("LIVEKIT_URL")
    lk_key = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")
    sip_trunk_id = os.getenv("SIP_TRUNK_ID")

    if not all([lk_url, lk_key, lk_secret, sip_trunk_id]):
        raise HTTPException(status_code=500, detail="Microservice environment is unconfigured.")

    clean_phone = payload.phone_number.replace("+", "").replace(" ", "")
    room_name = f"room_manjunath_{clean_phone}"

    pending_calls[room_name] = {
        "contact_name": payload.contact_name.strip(),
        "initial_message": payload.initial_message.strip()
    }

    log.info(f"Initiating call to {payload.contact_name} ({payload.phone_number})")

    try:
        lk_api = api.LiveKitAPI(lk_url, lk_key, lk_secret)
        await lk_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=payload.phone_number.strip(),
                room_name=room_name,
                participant_identity=f"sip_{clean_phone}"
            )
        )
        await lk_api.close()
        return {"status": "success", "message": f"Call dispatched for room: {room_name}"}
        
    except Exception as e:
        log.exception("Error executing LiveKit SIP caller dispatch")
        pending_calls.pop(room_name, None)
        raise HTTPException(status_code=500, detail=f"Telephony error: {str(e)}")


# ── POST-CALL SUMMARIZATION LOGIC ─────────────────────────────────────────────
async def summarize_and_webhook(history_text: str, contact_name: str):
    """Uses Groq to summarize the transcript and sends it back to app.py"""
    groq_key = os.getenv("SECONDARY_GROQ_API_KEY")
    # Provide the URL where your main app.py is hosted (or localhost for testing)
    main_app_url = os.getenv("MAIN_APP_URL", "http://localhost:8000")
    
    prompt = (
        f"Summarize the following phone call transcript between an AI assistant representing Manjunath "
        f"and a contact named {contact_name}.\n\n"
        f"Transcript:\n{history_text}\n\n"
        f"Keep the summary professional, concise, and focused on the outcome."
    )
    
    headers = {"Authorization": f"Bearer {groq_key}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        async with httpx.AsyncClient() as client:
            log.info("Generating call summary via Groq...")
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20.0)
            resp.raise_for_status()
            summary = resp.json()["choices"][0]["message"]["content"]
            log.info(f"Summary Generated: {summary}")
            
            # Send webhook to your main app to trigger the self-email
            webhook_payload = {"contact_name": contact_name, "summary": summary}
            log.info("Dispatching summary webhook to Main App...")
            wh_resp = await client.post(f"{main_app_url}/webhook/call-summary", json=webhook_payload, timeout=10.0)
            wh_resp.raise_for_status()
            log.info("Summary successfully sent to main app for emailing.")
    except Exception as e:
        log.error(f"Failed to generate or send summary: {e}")


# ── LIVEKIT REAL-TIME AI VOICE ENGINE ─────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    log.info(f"Phone audio pipeline active for room: {ctx.room.name}.")
    await ctx.connect()

    call_metadata = pending_calls.pop(ctx.room.name, {
        "contact_name": "there",
        "initial_message": "I have an automated update for you."
    })
    
    contact_name = call_metadata["contact_name"]
    initial_message = call_metadata["initial_message"]
    groq_key = os.getenv("SECONDARY_GROQ_API_KEY")

    system_instructions = (
        f"You are a helpful, professional AI voice assistant calling on behalf of Manjunath. "
        f"You are speaking directly to {contact_name}. "
        f"Keep responses concise. Your objective is to deliver this message: '{initial_message}'. "
        f"Handle any questions contextually."
    )

    agent = WorkerAgent(
        instructions=system_instructions,
        llm=openai.LLM(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", api_key=groq_key),
        stt=openai.STT(model="whisper-large-v3", base_url="https://api.groq.com/openai/v1", api_key=groq_key),
        tts=openai.TTS() 
    )

    await agent.start(ctx.room)
    await agent.say(f"Hello {contact_name}, I am an AI voice assistant calling on behalf of Manjunath. He wanted me to tell you that: {initial_message}")

    # Create an event flag to pause script execution until the user hangs up
    call_ended = asyncio.Event()

    @ctx.room.on("disconnected")
    def on_disconnected(*args, **kwargs):
        log.info(f"Call with {contact_name} disconnected. Triggering wrap-up.")
        call_ended.set()

    # Pause here and keep the agent alive until the phone hangs up
    await call_ended.wait()

    # The call is over! Extract the transcript for the summary.
    history_text = ""
    for msg in agent.chat_ctx.messages:
        # We skip the system prompt to save token usage
        if msg.role != "system":
            history_text += f"{msg.role.capitalize()}: {msg.content}\n"
            
    if history_text.strip():
        # Await the summarization webhook before letting the worker shut down
        await summarize_and_webhook(history_text, contact_name)


if __name__ == "__main__":
    web_port = int(os.getenv("PORT", 8001))
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": web_port, "log_level": "info"}, daemon=True).start()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))