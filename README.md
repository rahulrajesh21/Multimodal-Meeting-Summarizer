# Vela

**Agentic Meeting Intelligence Platform**

An open-source multimodal meeting summarization and automation platform that moves beyond passive transcription — capturing audio and visual streams, maintaining a persistent memory of organizational knowledge, and acting on it autonomously via an integrated AI agent.

---

## About the Project

Vela is not a transcription tool. It is a reasoning layer placed on top of your meetings.

While most tools hand you a wall of text after a call, Vela captures what was said, what was shown on screen, who said it, and what decisions were made — then acts on that information directly inside your existing workspace. Documents are drafted, sheets are updated, calendar context is applied, and Slack messages are dispatched, all without manual intervention.

It is designed to run locally. Your meeting data stays on your infrastructure. No vendor lock-in. No cloud dependency. If you stop using Vela, your Google Workspace and Slack remain exactly as you left them.

---

## Key Features

**Multimodal Extraction Pipeline**
Combines `faster-whisper` / `WhisperX` for accurate audio transcription and speaker diarization with robust visual modeling to detect screen shares and perform OCR on presentation slides.

**Temporal Knowledge Graph**
A persistent graph that maps entities — decisions, topics, projects — across multiple meetings, preserving context over time rather than treating each session in isolation.

**Agentic Automation via MCP**
Utilizes the Model Context Protocol (MCP) to communicate with external tools autonomously. The agent can read and draft Google Docs, manipulate Google Sheets, query your Google Calendar, and send messages or notifications in Slack.

**Warm Light Dashboard**
A Next.js user interface built on Cosmic Glassmorphism design principles for intuitive meeting review and live agent monitoring.

**Local-Priority LLM Inference**
Designed to run reasoning models locally via LM Studio to ensure data privacy without sacrificing capability.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend / Pipeline | Python, FastAPI, Uvicorn, SQLite, ChromaDB |
| Frontend / Client | React, Next.js, Tailwind CSS, Lucide Icons, Simple Icons |
| ML Models | Whisper, Sentence-Transformers, pyannote.audio, DistilBERT |
| Integration Layer | Model Context Protocol (MCP) |

---

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- FFmpeg (required for audio processing)
- A local LLM server such as [LM Studio](https://lmstudio.ai) running on the default port

### Backend Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install UI packages
npm install

# Start the Next.js dashboard
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

---

## Architectural Highlights

**Active Workflows**
Unlike standard transcription tools that produce static summaries, Vela connects via MCP directly to your Google Workspace and Slack, turning spoken decisions into dispatched tickets and structured project specifications in real time.

**Server-Sent Events (SSE)**
A streaming architecture connects the frontend directly to the agent's reasoning chain and tool usage as it happens, so you can observe every step the agent takes during a session.

**Hardware Optimization**
Tested and natively accelerated for Apple Silicon via MPS. Runs efficiently on consumer hardware without requiring a dedicated GPU server.

---

## License

Open source and available for educational and commercial use.
