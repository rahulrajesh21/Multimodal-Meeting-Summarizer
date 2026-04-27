# Vela - Agentic Meeting Intelligence Platform

Vela is a state-of-the-art multimodal meeting summarization and automation platform. It moves beyond passive transcription by capturing audio and visual streams, maintaining a persistent memory of organizational knowledge, and acting on it autonomously via an integrated AI Agent.

## 🌟 Key Features

- **Multimodal Extraction Pipeline**: Combines faster-whisper/WhisperX for accurate audio transcription and speaker diarization, with robust visual modeling to detect screen shares and perform OCR on presentation slides.
- **Temporal Knowledge Graph**: A persistent graph mapping entities (decisions, topics, projects) across multiple meetings to preserve context over time.
- **Agentic Automation (MCP)**: Utilizes the Model Context Protocol (MCP) to seamlessly communicate with external tools. The autonomous agent can read and draft Google Docs, manipulate Google Sheets, view your Google Calendar, and send messages or notifications in Slack.
- **Warm Light Dashboard**: A sleek, beautifully crafted Next.js user interface utilizing Cosmic Glassmorphism principles for intuitive meeting review and live agent monitoring.
- **Local Priority LLM Inference**: Designed to run reasoning models locally via LM Studio to ensure data privacy without sacrificing capability.

## 🛠 Tech Stack

- **Backend / Pipeline**: Python, FastAPI, Uvicorn, SQLite, ChromaDB
- **Frontend / Client**: React, Next.js, Tailwind CSS, Lucide Icons, Simple Icons
- **ML Models**: Whisper, Sentence-Transformers, pyannote.audio (for diarization), DistilBERT
- **Integration Layer**: Model Context Protocol (MCP)

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **FFmpeg** (required for deep audio processing)
- **Local LLM Server (e.g., LM Studio)** running on the default port

### Backend Setup

```bash
# Create and activate virtual environment
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

## 🧠 Architectural Highlights

1. **Active Workflows**: Unlike standard tools which leave you with a massive block of text, Vela connects via MCP directly to your Google Workspace and Slack, turning spoken decisions into dispatched tickets and structured project specs.
2. **Server-Sent Events (SSE)**: High-speed streaming architecture connects the frontend directly to the agent's real-time reasoning and tool usage chain. 
3. **Hardware Optimization**: Tested and accelerated natively for Apple Silicon (MPS).

## 📝 License
Open Source & Available for educational/commercial use.

