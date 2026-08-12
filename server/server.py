"""
Liquid Audio GUI – Hybrid local server
STT  : faster-whisper
LLM  : LM Studio local API at http://127.0.0.1:1234/v1/chat/completions
TTS  : edge-tts

Run:
    python server.py
Open:
    http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import edge_tts
import numpy as np
import requests

# ---------- config ----------
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "lfm2.5-audio-1.5b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
SAMPLE_RATE = 16000
CHANNELS = 1
HTTP_HOST = os.getenv("HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8765"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("liquid-audio-gui")

# ---------- state ----------
app = FastAPI(title="Liquid Audio GUI Server")
whisper_model: Optional[WhisperModel] = None


# ---------- STT ----------
def init_whisper() -> None:
    global whisper_model
    log.info("Loading Whisper model: %s (%s / %s)", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
    whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    log.info("Whisper ready.")


def transcribe_pcm16(pcm_bytes: bytes, sr: int = SAMPLE_RATE) -> str:
    assert whisper_model is not None
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio, language="en", beam_size=5)
    return "".join(seg.text for seg in segments).strip()


MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"

# ---------- LLM ----------
def lm_studio_chat(messages: list[dict]) -> str:
    if MOCK_LLM:
        return "This is a mocked AI reply so the audio path can be tested without LM Studio."
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }
    r = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------- TTS ----------
async def synthesize_edge(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


# ---------- WebSocket client handler ----------
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "session.created"})

    history: list[dict] = [{"role": "system", "content": "You are a helpful voice assistant. Reply concisely."}]

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "session.update":
                await ws.send_json({"type": "session.updated"})

            elif mtype == "input_audio_buffer.append":
                b64 = msg.get("audio") or ""
                if not b64:
                    continue
                pcm = base64.b64decode(b64)

                text = transcribe_pcm16(pcm)
                if not text:
                    await ws.send_json({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "",
                        "is_final": False,
                    })
                    continue

                await ws.send_json({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": text,
                    "is_final": True,
                })
                history.append({"role": "user", "content": text})

                reply = lm_studio_chat(history)
                history.append({"role": "assistant", "content": reply})

                wav = await synthesize_edge(reply)
                wav_b64 = base64.b64encode(wav).decode("ascii")

                await ws.send_json({
                    "type": "response.audio.delta",
                    "delta": wav_b64,
                    "transcript": reply,
                })
                await ws.send_json({"type": "response.done"})

            elif mtype == "input_audio_buffer.commit":
                await ws.send_json({"type": "input_audio_buffer.committed"})

            elif mtype == "response.create":
                await ws.send_json({"type": "response.created"})

            else:
                await ws.send_json({"type": "error", "error": f"unknown event: {mtype}"})

    except WebSocketDisconnect:
        log.info("Client disconnected.")
    except Exception as e:
        log.exception("WS error: %s", e)
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


@app.get("/")
def index() -> HTMLResponse:
    html_path = PROJECT_ROOT / "gui" / "index.html"
    log.info("Serving GUI from %s exists=%s", html_path, html_path.exists())
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def run() -> None:
    init_whisper()
    import uvicorn
    log.info("Server ready at http://%s:%s", HTTP_HOST, HTTP_PORT)
    uvicorn.run(app, host=HTTP_HOST, port=HTTP_PORT, log_level="info")


if __name__ == "__main__":
    run()
