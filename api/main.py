"""
Meeting Intelligence Platform — FastAPI Backend
Bridges the existing src/ pipeline to the end-user frontend.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── directories ───────────────────────────────────────────────────────────────
UPLOADS_DIR = ROOT / "temp_uploads"
DATA_DIR    = ROOT / "data"
MEMORY_DIR  = DATA_DIR / "temporal_memory"
JOBS_FILE   = DATA_DIR / "jobs.json"
ROLES_FILE  = DATA_DIR / "participants.json"

for d in [UPLOADS_DIR, DATA_DIR, MEMORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── shared singletons ─────────────────────────────────────────────────────────
text_analyzer     = None
temporal_memory   = None
participant_store = None

def get_text_analyzer():
    global text_analyzer
    if text_analyzer is None:
        from src.text_analysis import TextAnalyzer
        text_analyzer = TextAnalyzer()
    return text_analyzer

def get_temporal_memory():
    global temporal_memory
    if temporal_memory is None:
        from src.temporal_graph_memory import TemporalGraphMemory
        ta = get_text_analyzer()
        temporal_memory = TemporalGraphMemory(
            text_analyzer=ta,
            storage_path=str(MEMORY_DIR)
        )
    return temporal_memory

def get_participant_store():
    global participant_store
    if participant_store is None:
        from src.participant_store import ParticipantStore
        participant_store = ParticipantStore(data_dir=str(DATA_DIR))
    return participant_store

# ── job store (in-memory + persisted) ────────────────────────────────────────
_jobs: Dict[str, Dict] = {}

def _load_jobs():
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(_jobs, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save jobs: {e}")

_jobs = _load_jobs()

# ── job processing ────────────────────────────────────────────────────────────
_processing_semaphore = asyncio.Semaphore(1)   # one meeting at a time

async def process_meeting_job(job_id: str):
    """Full pipeline: transcribe → score → ingest → summarize."""
    async with _processing_semaphore:
        job = _jobs.get(job_id)
        if not job:
            return

        def _set(key, val):
            _jobs[job_id][key] = val
            _save_jobs()

        try:
            _set("status", "processing")
            _set("progress", 5)
            _set("stage", "Initializing models…")

            video_path = job["video_path"]
            participants = job.get("participants", [])
            meeting_title = job.get("title", "Untitled Meeting")

            # ── register participants ─────────────────────────────────────────
            pstore = get_participant_store()
            for p in participants:
                name = p.get("name", "Unknown")
                role = p.get("role", "Attendee")
                try:
                    pstore.add_participant(name=name, role=role,
                                          is_external=p.get("is_external", False),
                                          department=p.get("department", ""))
                except Exception:
                    pass

            _set("progress", 10)
            _set("stage", "Loading audio…")

            # ── load audio as numpy array (librosa → 16 kHz mono) ─────────────
            await asyncio.sleep(0)
            import numpy as np
            try:
                import librosa
                loop = asyncio.get_event_loop()
                audio_array, sr = await loop.run_in_executor(
                    None,
                    lambda: librosa.load(video_path, sr=16000, mono=True)
                )
                logger.info(f"Loaded audio: {len(audio_array)/16000:.1f}s @ 16kHz")
            except Exception as e:
                logger.warning(f"Audio load failed ({e}) — trying soundfile fallback")
                try:
                    import soundfile as sf
                    audio_array, sr = sf.read(video_path, dtype='float32', always_2d=False)
                    if sr != 16000:
                        import scipy.signal
                        audio_array = scipy.signal.resample(
                            audio_array, int(len(audio_array) * 16000 / sr)
                        ).astype('float32')
                    sr = 16000
                except Exception as e2:
                    logger.error(f"All audio loading failed: {e2}")
                    audio_array = None
                    sr = 16000

            _set("progress", 20)
            _set("stage", "Transcribing audio…")

            # ── transcription ─────────────────────────────────────────────────
            segments = []
            if audio_array is not None:
                try:
                    from src.live_transcription import LiveTranscriber
                    transcriber = LiveTranscriber()
                    loop = asyncio.get_event_loop()
                    segments = await loop.run_in_executor(
                        None,
                        lambda: list(transcriber.transcribe_full_audio(audio_array, sample_rate=sr))
                    )
                    # Filter out error segments
                    segments = [s for s in segments if not s.get('text','').startswith('Error:')]
                    logger.info(f"Transcribed {len(segments)} segments")
                except Exception as e:
                    logger.warning(f"Transcription failed ({e})")
                    segments = []

            # ── auto-map SPEAKER_XX labels → participant names ──────────────
            # Build ordered list of unique raw labels in first-appearance order
            seen_labels: list = []
            for seg in segments:
                lbl = seg.get("speaker") or "SPEAKER_00"
                if lbl not in seen_labels:
                    seen_labels.append(lbl)

            # Load existing custom map (user may have patched it) or build default
            existing_map: dict = job.get("speaker_map", {})
            speaker_map: dict = {}
            for i, lbl in enumerate(seen_labels):
                if lbl in existing_map:
                    speaker_map[lbl] = existing_map[lbl]
                elif i < len(participants):
                    speaker_map[lbl] = participants[i].get("name", lbl)
                else:
                    speaker_map[lbl] = lbl   # keep raw label as fallback

            _set("speaker_map", speaker_map)
            logger.info(f"Speaker map: {speaker_map}")

            # Apply map to all segments
            for seg in segments:
                raw = seg.get("speaker") or "SPEAKER_00"
                seg["speaker"] = speaker_map.get(raw, raw)

            transcript_text = "\n".join(
                f"{s.get('speaker', 'Speaker')}: {s.get('text', '')}"
                for s in segments if s.get("text", "").strip()
            )
            _set("transcript", transcript_text)
            _set("progress", 40)
            _set("stage", "Scoring segments…")

            # ── fusion scoring ────────────────────────────────────────────────
            from src.fusion_layer import FusionLayer
            ta = get_text_analyzer()
            fl = FusionLayer(text_analyzer=ta, temporal_memory=get_temporal_memory())

            seg_dicts = segments if segments else []
            if not seg_dicts and transcript_text:
                for i, line in enumerate(transcript_text.splitlines()):
                    if line.strip():
                        parts = line.split(":", 1)
                        seg_dicts.append({
                            "speaker": parts[0].strip() if len(parts) > 1 else "Unknown",
                            "text": parts[-1].strip(),
                            "start": i * 5.0,
                            "end": (i + 1) * 5.0,
                        })

            loop = asyncio.get_event_loop()
            scored = await loop.run_in_executor(
                None,
                lambda: fl.score_segments(seg_dicts, role="Attendee")
            )
            _set("progress", 65)
            _set("stage", "Ingesting into memory…")

            # ── temporal memory ingest ────────────────────────────────────────
            tm   = get_temporal_memory()
            meeting_id = tm.create_meeting(
                title=meeting_title,
                participants=[p.get("name") for p in participants],
                tags=job.get("tags", []),
            )
            _set("meeting_id", meeting_id)

            tm.ingest_meeting_results(meeting_id, scored, importance_threshold=0.35)
            tm.save()

            _set("progress", 80)
            _set("stage", "Generating summaries…")

            # ── per-participant summaries ──────────────────────────────────────
            summaries: Dict[str, str] = {}
            try:
                from src.llm_summarizer import LLMSummarizer
                llm = LLMSummarizer()

                from collections import defaultdict
                speaker_segs: Dict[str, list] = defaultdict(list)
                for seg in scored:
                    spk = getattr(seg, "speaker", None) or "Unknown"
                    speaker_segs[spk].append(seg)

                for spk, segs in speaker_segs.items():
                    profile = pstore.get_participant(spk)
                    role    = profile["role"] if profile else spk
                    top_segs = sorted(segs, key=lambda s: s.fused_score, reverse=True)[:7]
                    top_segs = sorted(top_segs, key=lambda s: s.start_time)
                    text_block = "\n".join(
                        f"[{s.start_time:.1f}s] {s.text}" for s in top_segs
                    )
                    if llm.is_ready and text_block.strip():
                        summaries[spk] = llm.summarize(text_block, role=role)
                    else:
                        summaries[spk] = "\n".join(
                            f"• {s.text[:120]}" for s in top_segs
                        )
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
                summaries = {}

            _set("summaries", summaries)
            _set("scored_count", len(scored))
            _set("progress", 100)
            _set("stage", "Complete")
            _set("status", "done")
            _set("completed_at", datetime.now().isoformat())

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            _set("status", "error")
            _set("error", str(e))
            _set("stage", f"Error: {e}")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Meeting Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  MEETINGS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/meetings/upload")
async def upload_meeting(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    title: str = Form("Untitled Meeting"),
    participants: str = Form("[]"),   # JSON array
    tags: str = Form("[]"),
):
    """Upload a video file + participants. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]

    # Save video
    ext = Path(video.filename).suffix or ".mp4"
    video_path = UPLOADS_DIR / f"{job_id}{ext}"
    async with aiofiles.open(video_path, "wb") as f:
        content = await video.read()
        await f.write(content)

    try:
        parsed_participants = json.loads(participants)
    except Exception:
        parsed_participants = []

    try:
        parsed_tags = json.loads(tags)
    except Exception:
        parsed_tags = []

    job = {
        "job_id":      job_id,
        "title":       title,
        "video_path":  str(video_path),
        "video_filename": video.filename,
        "participants":parsed_participants,
        "tags":        parsed_tags,
        "status":      "queued",
        "progress":    0,
        "stage":       "Queued",
        "meeting_id":  None,
        "summaries":   {},
        "transcript":  "",
        "scored_count":0,
        "created_at":  datetime.now().isoformat(),
        "completed_at":None,
        "error":       None,
    }
    _jobs[job_id] = job
    _save_jobs()

    background_tasks.add_task(process_meeting_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/meetings")
def list_meetings():
    """Return all jobs (queued, processing, done) sorted newest first."""
    jobs = sorted(_jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)
    # Merge metadata from temporal memory if done
    tm = get_temporal_memory()
    result = []
    for j in jobs:
        item = {k: v for k, v in j.items() if k != "transcript"}  # omit big fields
        # attach temporal memory stats if available
        mid = j.get("meeting_id")
        if mid and j["status"] == "done":
            item["events"]  = len(tm.events_by_meeting.get(mid, []))
            item["topics"]  = sum(
                1 for eid in tm.entities_by_type.get("topic", [])
                if any(tm.events[ev].meeting_id == mid
                       for ev in tm.events_by_entity.get(eid, [])
                       if ev in tm.events)
            )
        else:
            item["events"] = 0
            item["topics"] = 0
        result.append(item)
    return result


@app.get("/api/meetings/{job_id}")
def get_meeting(job_id: str):
    """Full meeting detail including summaries and scored_segments info."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Meeting not found")

    result = dict(job)
    mid = job.get("meeting_id")
    if mid:
        tm = get_temporal_memory()
        events = [tm.events[eid].to_dict()
                  for eid in tm.events_by_meeting.get(mid, [])\
                  if eid in tm.events]
        result["graph_events"] = events[:100]
    return result


@app.patch("/api/meetings/{job_id}/speakers")
async def update_speaker_map(job_id: str, body: dict, background_tasks: BackgroundTasks):
    """
    Save a corrected speaker map and re-run scoring/summarization.
    Body: { "speaker_map": { "SPEAKER_00": "Alice", "SPEAKER_01": "Bob", ... } }
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") == "processing":
        raise HTTPException(409, "Job is already processing")
    new_map = body.get("speaker_map", {})
    if not new_map:
        raise HTTPException(400, "speaker_map is required")
    _jobs[job_id]["speaker_map"] = new_map
    # Reset so reprocess picks up the new map
    _jobs[job_id].update({
        "status": "queued", "progress": 0, "stage": "Re-queued with new speaker map",
        "transcript": "", "scored_count": 0, "summaries": {},
        "meeting_id": None, "error": None, "completed_at": None,
    })
    _save_jobs()
    background_tasks.add_task(process_meeting_job, job_id)
    return {"job_id": job_id, "status": "queued", "speaker_map": new_map}




@app.get("/api/meetings/{job_id}/video")
def stream_video(job_id: str):
    """Serve the uploaded video file."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404, "Video file not found")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=job.get("video_filename", "meeting.mp4"))


@app.post("/api/meetings/{job_id}/reprocess")
async def reprocess_meeting(job_id: str, background_tasks: BackgroundTasks):
    """Re-run the full pipeline for an existing job (useful after transcription errors)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "processing":
        raise HTTPException(409, "Job is already processing")
    # Reset state
    _jobs[job_id].update({
        "status": "queued", "progress": 0, "stage": "Re-queued",
        "transcript": "", "scored_count": 0, "summaries": {},
        "meeting_id": None, "error": None, "completed_at": None,
    })
    _save_jobs()
    background_tasks.add_task(process_meeting_job, job_id)
    return {"job_id": job_id, "status": "queued"}


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-MEETING THREADS / GRAPH
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/threads")
def get_threads(min_meetings: int = 2, threshold: float = 0.50):
    """Return cross-meeting entity threads for graph visualization."""
    tm = get_temporal_memory()
    threads = tm.find_cross_meeting_threads(
        min_meetings=min_meetings,
        similarity_threshold=threshold,
    )
    return {"threads": threads, "total": len(threads)}


@app.get("/api/graph")
def get_graph():
    """
    Returns nodes + links for force-directed graph.
    Nodes: meetings (blue) + entities (purple).
    Links: entity ↔ meeting when they share events.
    """
    tm = get_temporal_memory()
    nodes = []
    links = []
    seen_node_ids = set()

    # Meeting nodes
    for mid, m in tm.meetings.items():
        nodes.append({"id": f"m_{mid}", "label": m["title"],
                      "type": "meeting", "date": m.get("date", "")})
        seen_node_ids.add(f"m_{mid}")

    # Entity nodes + links
    for ent_id, ent in tm.entities.items():
        evt_ids = tm.events_by_entity.get(ent_id, [])
        meeting_ids = {tm.events[eid].meeting_id
                       for eid in evt_ids if eid in tm.events}
        if not meeting_ids:
            continue

        node_id = f"e_{ent_id}"
        nodes.append({
            "id":    node_id,
            "label": ent.canonical_name[:40],
            "type":  ent.type,
            "mentions": ent.mention_count,
            "recurrence": round(ent.recurrence_score, 3),
            "unresolved": round(ent.unresolved_score, 3),
        })

        for mid in meeting_ids:
            m_node = f"m_{mid}"
            if m_node in seen_node_ids:
                links.append({"source": node_id, "target": m_node,
                              "value": ent.mention_count})

    return {"nodes": nodes, "links": links}


# ═══════════════════════════════════════════════════════════════════════════
#  ROLES / PARTICIPANTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/roles")
def list_roles():
    pstore = get_participant_store()
    return {"participants": pstore.list_participants()}


@app.post("/api/roles")
def add_role(body: dict):
    """
    Body: { name, role, department?, is_external?, authority? }
    """
    name = body.get("name", "").strip()
    role = body.get("role", "").strip()
    if not name or not role:
        raise HTTPException(400, "name and role are required")

    pstore = get_participant_store()
    profile = pstore.add_participant(
        name=name,
        role=role,
        department=body.get("department", ""),
        is_external=body.get("is_external", False),
    )
    return profile


@app.delete("/api/roles/{name}")
def delete_role(name: str):
    pstore = get_participant_store()
    slug = pstore._slugify(name)
    if slug not in pstore._profiles:
        raise HTTPException(404, "Participant not found")
    del pstore._profiles[slug]
    pstore._save()
    return {"deleted": name}


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "jobs": len(_jobs),
    }
