# Multimodal Meeting Summarizer

**Neural multimodal fusion pipeline for automated meeting transcription, speaker diarization, and personalized highlight extraction.**

Combines acoustic prosody, contextual visual cues, and semantic text representations into a unified scoring model to identify key discussions, decisions, and action items from recorded or live meetings.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │        Meeting Input Stream          │
                      │       (Audio + Video / Slides)       │
                      └──────────────────┬───────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  Acoustic Stream │           │  Textual Stream  │           │  Visual Stream   │
├──────────────────┤           ├──────────────────┤           ├──────────────────┤
│ • faster-whisper │           │ • DistilBERT     │           │ • ResNet-50      │
│ • pyannote audio │           │ • Sentence-BERT  │           │ • Screen/Slide   │
│ • MFCC prosody   │           │ • Sentiment & NER│           │   detection+OCR  │
└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘
         │                               │                               │
         └───────────────────────┬───────┴───────────────────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │    Multimodal Fusion Layer    │
                 │      (`fusion_model.pth`)     │
                 │   Cross-modal attention &     │
                 │   importance score ranking    │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     Temporal Knowledge Graph    │   │     Actionable Highlights       │
│  4-layer entity & decision map  │   │  Automated video clips, notes,  │
│  persisting context over time   │   │  and structured action items    │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## Key Components

### 1. Acoustic & Diarization Pipeline
- **Transcription**: Accelerated inference using `faster-whisper` across multiple model tiers (`base`, `small`, `medium`).
- **Speaker Diarization**: Turn-level speaker tracking powered by `pyannote.audio`.
- **Prosodic Feature Extraction**: Extracts MFCCs, energy envelopes, and pitch contours to capture vocal emphasis and urgency during discussions.

### 2. Semantic & Visual Context
- **Textual Embeddings**: Token- and segment-level embeddings generated with DistilBERT and Sentence-Transformers to score semantic density and identify intent.
- **Visual Scene Analysis**: Frame-sampling pipeline with a fine-tuned `ResNet-50` backbone and OCR to incorporate slide presentations, screen shares, and whiteboard transitions into the timeline.

### 3. Neural Multimodal Fusion (`src/fusion_layer.py`)
- Evaluates segment relevance by combining semantic significance, vocal emphasis, and visual changes.
- Supports weighted linear, multiplicative, and gated cross-attention fusion strategies with per-speaker role calibration.

### 4. 4-Layer Temporal Knowledge Graph (`src/temporal_graph_memory.py`)
- **Layer 0 (Ingestion)**: Segment filtering and backchannel suppression.
- **Layer 1 (Event Extraction)**: Structured extraction of decisions, blockers, risks, and milestones.
- **Layer 2 (Canonicalization)**: Dual-pass entity resolution combining cosine similarity over embeddings with Levenshtein lexical matching.
- **Layer 3 (Temporal Graph)**: Inter-meeting event graph linking topics and participants across recurring syncs.

---

## Repository Structure

```
.
├── api/                    # FastAPI backend endpoints & streaming handlers
│   └── main.py
├── src/                    # Core processing modules
│   ├── audio_analysis.py   # MFCC extraction & acoustic prosody
│   ├── live_transcription.py # Whisper transcription & diarization
│   ├── text_analysis.py    # DistilBERT embeddings & NER
│   ├── visual_analysis.py  # ResNet-50 visual features & slide OCR
│   ├── fusion_layer.py     # Multimodal fusion neural network
│   ├── temporal_graph_memory.py # 4-layer temporal knowledge graph
│   └── video_processing.py # Automated highlight clipping via MoviePy
├── tools/                  # Distillation datasets & evaluation scripts
├── tests/                  # Unit and integration test suites
├── evaluate_ami_results.py # Quantitative evaluation on AMI Corpus
├── calculate_metrics.py    # Precision, recall, and ROUGE scoring
└── requirements.txt        # Python dependencies
```

---

## Evaluation & Benchmarks

The fusion model and extraction pipeline are evaluated on the **AMI Meeting Corpus** (`load_ami_corpus.py`):

```bash
# Run quantitative benchmark on AMI meeting test partitions
python evaluate_ami_results.py

# Compute summarization and alignment metrics
python calculate_metrics.py
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- FFmpeg (required for audio demuxing and clip generation)
- Node.js 18+ (optional, for Next.js web client)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/rahulrajesh21/Multimodal-Meeting-Summarizer.git
cd Multimodal-Meeting-Summarizer

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Backend Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Run Streamlit Inspection Dashboard

```bash
streamlit run streamlit_app.py
```

### 4. Run Test Suite

```bash
pytest tests/
```

---

## Hardware Optimization

- **Apple Silicon (MPS)**: PyTorch MPS acceleration is natively supported and enabled by default for local inference on macOS.
- **CUDA**: Automatically detected for GPU-accelerated transcription and feature extraction.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
