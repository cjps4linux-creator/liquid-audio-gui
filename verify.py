"""Liquid Audio GUI – Quick SMOKE test

Verifies the local hybrid voice pieces externally:
- faster-whisper import + CPU probe
- LM Studio local API reachable
- edge-tts writes a WAV
- server HTML route serves

Usage:
    python verify.py
    LM_STUDIO_URL=http://127.0.0.1:1234/v1/chat/completions python verify.py
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import requests
import edge_tts


def banner(title: str) -> None:
    print(f"\n== {title} ==")


def check_whisper() -> bool:
    banner("faster-whisper CPU probe")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio[int(sr * 0.8):] = 0.02 * np.random.randn(int(sr * 0.2)).astype(np.float32)
    t0 = time.time()
    segments, info = model.transcribe(audio, language="en", beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    dt = time.time() - t0
    print(f"transcribe={dt:.2f}s | text={text!r}")
    return dt > 0


def check_lm_studio() -> bool:
    banner("LM Studio")
    base = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
    models_url = base.replace("/chat/completions", "/models")
    m = requests.get(models_url, timeout=20)
    print("models status", m.status_code)
    m.raise_for_status()
    payload = {
        "model": "lfm2.5-audio-1.5b",
        "messages": [{"role": "user", "content": "Reply with exactly: verification successful"}],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(base, json=payload, timeout=120)
    print("chat status", r.status_code, "in", round(time.time() - __import__('time').time(), 2), "s")
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    print("reply=", repr(text))
    print("model=", data.get("model"))
    return "verification successful" in text


def check_tts() -> bool:
    banner("edge-tts")
    out = Path(tempfile.gettempdir()) / "liquid_audio_gui_tts_verify.wav"
    import asyncio

    async def _go():
        communicate = edge_tts.Communicate("Athena audio check.", "en-US-JennyNeural")
        with out.open("wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

    t0 = time.time()
    asyncio.run(_go())
    dt = time.time() - t0
    size = out.stat().st_size if out.exists() else 0
    print(f"wrote={out} size={size} time={dt:.2f}s")
    return size > 0


def check_server_route() -> bool:
    banner("server route")
    url = "http://127.0.0.1:8765/"
    try:
        r = requests.get(url, timeout=20)
    except requests.ConnectionError:
        print("FAIL server not running at", url)
        return False
    print("status", r.status_code, "len", len(r.text))
    return r.status_code == 200 and "<!doctype html>" in r.text.lower()


def main() -> int:
    results: List[bool] = []
    try:
        results.append(check_whisper())
    except Exception as e:
        print("whisper failed:", repr(e))
        results.append(False)
    try:
        results.append(check_lm_studio())
    except Exception as e:
        print("lm studio failed:", repr(e))
        results.append(False)
    try:
        results.append(check_tts())
    except Exception as e:
        print("tts failed:", repr(e))
        results.append(False)
    results.append(check_server_route())

    banner("RESULTS")
    for name, ok in zip(["whisper", "lm_studio", "tts", "server_route"], results):
        print(f" - {name}: {'ok' if ok else 'FAIL'}")
    return 0 if all(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
