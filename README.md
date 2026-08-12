# Liquid Audio GUI

**Hybrid local voice chat server — STT + LLM + TTS in one FastAPI app.**

Built by Conrad CJ Wilson. Runs entirely on CPU: faster-whisper for speech-to-text, LM Studio for LLM inference, and edge-tts for text-to-speech. No cloud APIs, no GPU required.

## What It Does

| Component | Tool | Model |
|---|---|---|
| Speech-to-Text | faster-whisper | `small` (CPU, int8) |
| Language Model | LM Studio local API | `lfm2.5-audio-1.5b` |
| Text-to-Speech | edge-tts | `en-US-JennyNeural` |

WebSocket-based real-time audio chat with a browser GUI.

## Architecture

```
Browser GUI (index.html)
    │  WebSocket
    ▼
FastAPI Server (server/server.py)
    ├── faster-whisper  →  STT (speech → text)
    ├── LM Studio API   →  LLM (text → response)
    └── edge-tts        →  TTS (response → audio)
```

## Quick Start

```bash
# Clone
git clone https://github.com/cjps4linux-creator/liquid-audio-gui.git
cd liquid-audio-gui

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Set LM_STUDIO_URL, WHISPER_MODEL, TTS_VOICE as needed

# Ensure LM Studio is running with lfm2.5-audio-1.5b at http://127.0.0.1:1234

# Run
python server/server.py

# Open browser
http://localhost:8765
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1/chat/completions` | LM Studio API endpoint |
| `LM_STUDIO_MODEL` | `lfm2.5-audio-1.5b` | Model identifier in LM Studio |
| `WHISPER_MODEL` | `small` | faster-whisper model size |
| `WHISPER_DEVICE` | `cpu` | Device: `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `int8` | Quantization: `int8`, `float16` |
| `TTS_VOICE` | `en-US-JennyNeural` | Edge TTS voice ID |
| `HTTP_HOST` | `127.0.0.1` | Server bind address |
| `HTTP_PORT` | `8765` | Server port |

## Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running with LFM2.5-Audio-1.5B model loaded
- `pip install -r requirements.txt`

## Requirements

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
faster-whisper>=1.0.0
edge-tts>=6.1.0
sounddevice>=0.4.0
soundfile>=0.12.0
numpy>=1.24.0
requests>=2.31.0
pydub>=0.25.0
```

## Security

- **No API keys in code** — LM Studio runs locally, no cloud auth needed
- Server binds to `127.0.0.1` by default (localhost only)
- No data leaves the machine
- `.env` is gitignored

## Performance Notes

- CPU-only operation tested on Intel i7-1185G7
- faster-whisper `small` model: ~1-2s transcription latency
- LM Studio with `int8` quantization keeps RAM under 4GB
- edge-tts streams audio chunks for low perceived latency

## Roadmap

- [ ] Confirm LM Studio audio streaming endpoints
- [ ] Complete GUI wiring for multi-turn conversation
- [ ] Add conversation history / session memory
- [ ] Support for additional Whisper model sizes

## License

MIT — see [LICENSE](LICENSE)

## Author

Conrad CJ Wilson — [@cjps4linux-creator](https://github.com/cjps4linux-creator)
