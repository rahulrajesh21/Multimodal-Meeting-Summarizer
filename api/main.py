"""
Meeting Intelligence Platform — FastAPI Backend
Bridges the existing src/ pipeline to the end-user frontend.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from contextlib import AsyncExitStack, asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

# ── Teams Media Server base URL ───────────────────────────────────────────
TEAMS_SERVER_URL = os.environ.get("TEAMS_SERVER_URL", "http://localhost:8001")

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
                        tm_instance = get_temporal_memory()
                        summaries[spk] = llm.summarize(text_block, role=role, scored_segments=top_segs, tm=tm_instance)
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

mcp_session: ClientSession = None
mcp_tools_cache = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_session, mcp_tools_cache
    
    # NOTE: Set up your MCP Server connection parameters here.
    # We are using Google Sheets as requested. Make sure `npx` is available and 
    # the proper Google credentials environment variables are passed when starting the API.
    server_params = StdioServerParameters(command="npx", args=["-y", "mcp-google-sheets"])
    
    async with AsyncExitStack() as stack:
        try:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            mcp_session = session
            
            # Fetch and cache MCP tools in OpenAI format
            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                mcp_tools_cache.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                })
            logger.info(f"MCP Server initialized successfully. Loaded {len(mcp_tools_cache)} tools.")
        except Exception as e:
            logger.warning(f"Failed to start MCP Client/Server: {e}")
        yield

app = FastAPI(title="Meeting Intelligence API", version="1.0.0", lifespan=lifespan)

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

@app.post("/api/meetings/{job_id}/extract-video")
async def extract_meeting_video(job_id: str, body: dict):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Meeting not found")
        
    speaker_filter = body.get("speaker", "all")
    topic_filter = body.get("topic", "all")
    
    mid = job.get("meeting_id")
    if not mid:
        raise HTTPException(400, "Meeting not fully processed yet")
        
    tm = get_temporal_memory()
    events = [tm.events[eid] for eid in tm.events_by_meeting.get(mid, []) if eid in tm.events]
    
    # Filter
    if speaker_filter != 'all':
        events = [e for e in events if getattr(e, 'speaker', None) == speaker_filter]
    if topic_filter != 'all':
        events = [e for e in events if getattr(e, 'event_type', None) == topic_filter]
        
    if not events:
        raise HTTPException(400, "No highlighted events match the current filter")
        
    time_ranges = []
    # Build a rough time range.
    for e in events:
        start = getattr(e, 'start_time', 0.0)
        end = getattr(e, 'end_time', start + 6.0)
        time_ranges.append((start, end))
        
    try:
        from src.video_processing import VideoSummarizer
        class DummyScorer:
            def score_sentence(self, text, role): return 1.0
        
        summarizer = VideoSummarizer(highlight_scorer=DummyScorer())
        
        # Merge overlapping time ranges
        merged_ranges = []
        time_ranges.sort(key=lambda x: x[0])
        for r in time_ranges:
            if not merged_ranges:
                merged_ranges.append(r)
            else:
                last_start, last_end = merged_ranges[-1]
                if r[0] <= last_end + 2.0:
                    merged_ranges[-1] = (last_start, max(last_end, r[1]))
                else:
                    merged_ranges.append(r)
                    
        # Determine output file
        exports_dir = ROOT / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"highlight_{job_id}_{int(time.time())}.mp4"
        output_path = exports_dir / filename
        
        # Get video source
        video_src = job.get("video_path")
            
        if not video_src or not Path(video_src).exists():
            raise HTTPException(400, f"Source video not found on disk at {video_src}")
            
        status = summarizer.create_summary_video(str(video_src), merged_ranges, str(output_path))
        
        if "No valid clips" in status or "Error" in status:
             raise HTTPException(500, f"Failed to generate video: {status}")
             
        return {"filename": filename}
    except Exception as e:
        logger.exception(f"Extract video error: {e}")
        raise HTTPException(500, f"Failed to extract video: {e}")

@app.get("/api/meetings/{job_id}/download-video/{filename}")
def download_extracted_video(job_id: str, filename: str):
    path = ROOT / "exports" / filename
    if not path.exists():
        raise HTTPException(404, "Video file not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)



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
#  PROCESS TEAMS MEETING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_vtt_to_segments(vtt_text: str) -> List[Dict]:
    """Parse VTT text into segment dicts with speaker, text, start, end."""
    segments = []
    blocks = re.split(r"\n{2,}", vtt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue
        parts = ts_line.split("-->")
        start_raw = parts[0].strip()
        end_raw = parts[1].strip().split()[0]

        # Convert VTT timestamp to seconds
        def _vtt_to_sec(ts: str) -> float:
            p = ts.replace(",", ".").split(":")
            p = list(reversed(p))
            return (float(p[0]) if len(p) > 0 else 0) \
                 + (float(p[1]) * 60 if len(p) > 1 else 0) \
                 + (float(p[2]) * 3600 if len(p) > 2 else 0)

        text_lines = [l for l in lines if "-->" not in l
                      and l != "WEBVTT" and not l.isdigit()]
        full_text = " ".join(text_lines)

        # Extract speaker from <v SpeakerName>text
        speaker_m = re.match(r"<v\s+([^>]+)>(.*)", full_text)
        if speaker_m:
            speaker = speaker_m.group(1).strip()
            text = re.sub(r"<[^>]+>", "", speaker_m.group(2)).strip()
        else:
            speaker = "Unknown"
            text = re.sub(r"<[^>]+>", "", full_text).strip()

        if text:
            segments.append({
                "speaker": speaker,
                "text": text,
                "start": _vtt_to_sec(start_raw),
                "end": _vtt_to_sec(end_raw),
            })
    return segments


async def process_teams_job(job_id: str):
    """Pipeline for Teams meetings: uses pre-existing VTT transcript instead of Whisper."""
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
            _set("stage", "Fetching data from Teams server…")

            teams_mid = job["teams_meeting_id"]
            transcript_id = job.get("teams_transcript_id")
            recording_id = job.get("teams_recording_id")
            meeting_title = job.get("title", "Untitled Meeting")

            async with httpx.AsyncClient(timeout=120.0) as client:
                # 1. Fetch meeting metadata
                r = await client.get(f"{TEAMS_SERVER_URL}/v1.0/me/onlineMeetings/{teams_mid}")
                if r.status_code != 200:
                    raise Exception(f"Teams meeting not found: {teams_mid}")
                teams_meeting = r.json()

                # Auto-select first transcript/recording if not specified
                transcripts = teams_meeting.get("transcripts", [])
                recordings = teams_meeting.get("recordings", [])

                if not transcript_id and transcripts:
                    transcript_id = transcripts[0]["id"]
                if not recording_id and recordings:
                    recording_id = recordings[0]["id"]

                if not transcript_id:
                    raise Exception("No transcript available for this meeting")

                _set("progress", 10)
                _set("stage", "Downloading transcript…")

                # 2. Fetch VTT transcript content
                r = await client.get(
                    f"{TEAMS_SERVER_URL}/v1.0/me/onlineMeetings/{teams_mid}"
                    f"/transcripts/{transcript_id}/content"
                )
                if r.status_code != 200:
                    raise Exception("Failed to download transcript from Teams server")
                vtt_text = r.text

                # 3. Parse VTT into segments
                segments = _parse_vtt_to_segments(vtt_text)
                if not segments:
                    raise Exception("VTT transcript is empty or could not be parsed")
                logger.info(f"Parsed {len(segments)} segments from VTT")

                # 4. Download recording (for audio-based tonal analysis)
                audio_array = None
                video_path = None
                if recording_id:
                    _set("progress", 15)
                    _set("stage", "Downloading recording…")
                    r = await client.get(
                        f"{TEAMS_SERVER_URL}/v1.0/me/onlineMeetings/{teams_mid}"
                        f"/recordings/{recording_id}/content"
                    )
                    if r.status_code == 200:
                        ext = ".mp4"
                        video_path = UPLOADS_DIR / f"{job_id}{ext}"
                        async with aiofiles.open(video_path, "wb") as f:
                            await f.write(r.content)
                        _set("video_path", str(video_path))
                        logger.info(f"Downloaded recording to {video_path}")

            _set("progress", 20)
            _set("stage", "Loading audio for tonal analysis…")

            # 5. Load audio from recording if available
            if video_path and Path(video_path).exists():
                try:
                    import numpy as np
                    import librosa
                    loop = asyncio.get_event_loop()
                    audio_array, sr = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(str(video_path), sr=16000, mono=True)
                    )
                    logger.info(f"Loaded audio: {len(audio_array)/16000:.1f}s @ 16kHz")
                except Exception as e:
                    logger.warning(f"Audio load failed ({e}) — tonal scoring disabled")
                    audio_array = None

            # Build transcript text for summaries
            transcript_text = "\n".join(
                f"{s['speaker']}: {s['text']}" for s in segments if s["text"].strip()
            )
            _set("transcript", transcript_text)

            # Collect unique speakers as participants
            seen_speakers = []
            for seg in segments:
                if seg["speaker"] not in seen_speakers:
                    seen_speakers.append(seg["speaker"])

            # Register participants and store in job for frontend
            pstore = get_participant_store()
            job_participants = []
            for name in seen_speakers:
                try:
                    pstore.add_participant(name=name, role="Attendee")
                except Exception:
                    pass
                job_participants.append({"name": name, "role": "Attendee"})
            _set("participants", job_participants)

            # Store video filename for the detail page
            if video_path:
                _set("video_filename", Path(video_path).name)

            _set("progress", 35)
            _set("stage", "Analyzing video frames…")

            # 5b. Visual analysis (screen sharing detection)
            visual_context = []
            if video_path and Path(video_path).exists():
                try:
                    from src.visual_analysis import VisualAnalyzer
                    va = VisualAnalyzer()
                    loop_va = asyncio.get_event_loop()
                    visual_context = await loop_va.run_in_executor(
                        None, lambda: va.analyze_video_context(str(video_path))
                    )
                    _set("visual_context_count", len(visual_context))
                    _set("screen_share_frames", sum(1 for v in visual_context if v.get('is_slide')))
                    logger.info(f"Visual analysis: {len(visual_context)} frames, "
                                f"{sum(1 for v in visual_context if v.get('is_slide'))} screen-sharing")
                except Exception as e:
                    logger.warning(f"Visual analysis failed (continuing without): {e}")

            _set("progress", 45)
            _set("stage", "Scoring segments…")

            # 6. Fusion scoring
            from src.fusion_layer import FusionLayer
            ta = get_text_analyzer()
            fl = FusionLayer(text_analyzer=ta, temporal_memory=get_temporal_memory())

            loop = asyncio.get_event_loop()
            scored = await loop.run_in_executor(
                None,
                lambda: fl.score_segments(segments, role="Attendee",
                                          audio_data=audio_array if audio_array is not None else None,
                                          sample_rate=16000)
            )

            # 6b. Map visual context to scored segments by timestamp
            if visual_context:
                for seg in scored:
                    # Find the closest visual frame to this segment's midpoint
                    seg_mid = (seg.start_time + seg.end_time) / 2
                    closest = min(visual_context, key=lambda v: abs(v['timestamp'] - seg_mid))
                    if abs(closest['timestamp'] - seg_mid) < 10.0:  # Within 10s
                        seg.is_screen_sharing = closest.get('is_slide', False)
                        seg.screen_share_confidence = closest.get('context_confidence', 0.0)
                        seg.ocr_text = closest.get('ocr_text', '')
                        if closest.get('embedding') is not None:
                            seg.visual_embedding = closest['embedding']

            _set("progress", 60)
            _set("stage", "Ingesting into temporal memory…")

            # 7. Temporal memory ingest
            tm = get_temporal_memory()
            memory_meeting_id = tm.create_meeting(
                title=meeting_title,
                participants=seen_speakers,
                tags=job.get("tags", []),
            )
            _set("meeting_id", memory_meeting_id)

            tm.ingest_meeting_results(memory_meeting_id, scored, importance_threshold=0.1)
            logger.info(
                f"Ingested meeting {memory_meeting_id}: "
                f"{len(tm.events_by_meeting.get(memory_meeting_id, []))} events, "
                f"score range: {min((getattr(s, 'fused_score', 0) for s in scored), default=0):.3f}"
                f"-{max((getattr(s, 'fused_score', 0) for s in scored), default=0):.3f}"
            )
            tm.save()

            _set("progress", 75)
            _set("stage", "Generating summaries…")

            # 8. Per-participant LLM summaries
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
                    role = profile["role"] if profile else "Attendee"
                    top_segs = sorted(segs, key=lambda s: s.fused_score, reverse=True)[:7]
                    top_segs = sorted(top_segs, key=lambda s: s.start_time)
                    text_block = "\n".join(
                        f"[{s.start_time:.1f}s] {s.text}" for s in top_segs
                    )
                    if llm.is_ready and text_block.strip():
                        tm_instance = get_temporal_memory()
                        summaries[spk] = llm.summarize(text_block, role=role, scored_segments=top_segs, tm=tm_instance)
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
            logger.exception(f"Teams job {job_id} failed: {e}")
            _set("status", "error")
            _set("error", str(e))
            _set("stage", f"Error: {e}")


@app.post("/api/meetings/process-teams")
async def process_teams_meeting(body: dict, background_tasks: BackgroundTasks):
    """
    Process a meeting from the Teams Media Server.
    Body: { teams_meeting_id, transcript_id?, recording_id? }
    """
    teams_mid = body.get("teams_meeting_id", "").strip()
    if not teams_mid:
        raise HTTPException(400, "teams_meeting_id is required")

    # Fetch meeting title from Teams server
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{TEAMS_SERVER_URL}/v1.0/me/onlineMeetings/{teams_mid}")
            if r.status_code != 200:
                raise HTTPException(404, "Teams meeting not found")
            teams_data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Cannot reach Teams server: {e}")

    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id":        job_id,
        "title":         teams_data.get("subject", "Untitled Teams Meeting"),
        "video_path":    "",
        "video_filename": "",
        "participants":  [],
        "tags":          [],
        "status":        "queued",
        "progress":      0,
        "stage":         "Queued — Teams meeting",
        "meeting_id":    None,
        "summaries":     {},
        "transcript":    "",
        "scored_count":  0,
        "created_at":    datetime.now().isoformat(),
        "completed_at":  None,
        "error":         None,
        # Teams-specific fields
        "teams_meeting_id":    teams_mid,
        "teams_transcript_id": body.get("transcript_id", ""),
        "teams_recording_id":  body.get("recording_id", ""),
    }
    _jobs[job_id] = job
    _save_jobs()

    background_tasks.add_task(process_teams_job, job_id)
    return {"job_id": job_id, "status": "queued"}


# ═══════════════════════════════════════════════════════════════════════════
#  MEETING CHAT
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/meetings/{job_id}/chat")
async def chat_with_meeting(job_id: str, body: dict):
    """
    Chat with a meeting using LLM.  Sends full meeting context to LM Studio.
    Body: { "message": "...", "history": [...] }
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Meeting not found")
    if job.get("status") != "done":
        raise HTTPException(400, "Meeting is still processing")

    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        raise HTTPException(400, "message is required")

    history = body.get("history", [])[-10:]  # Keep last 10 exchanges

    # ── Build meeting context ──────────────────────────────────────────────
    ctx_parts: list[str] = []

    # 1. Metadata
    ctx_parts.append(f"Meeting Title: {job.get('title', 'Untitled')}")
    ctx_parts.append(f"Date: {job.get('created_at', 'Unknown')}")
    participants = job.get("participants", [])
    if participants:
        names = ", ".join(
            p.get("name", "?") + f" ({p.get('role', 'Attendee')})"
            for p in participants
        )
        ctx_parts.append(f"Participants: {names}")

    # 2. Per-speaker summaries (compact but rich)
    summaries = job.get("summaries", {})
    if summaries:
        ctx_parts.append("\n--- SPEAKER SUMMARIES ---")
        for speaker, summary in summaries.items():
            ctx_parts.append(f"{speaker}:\n{summary}")

    # 3. Graph events (decisions, problems, ideas with timestamps)
    mid = job.get("meeting_id")
    if mid:
        tm = get_temporal_memory()
        event_ids = tm.events_by_meeting.get(mid, [])
        if event_ids:
            ctx_parts.append("\n--- KEY EVENTS ---")
            for eid in event_ids[:50]:
                ev = tm.events.get(eid)
                if ev:
                    line = f"[{ev.start_time:.0f}s] [{ev.event_type}] {ev.speaker}: {ev.summary}"
                    if ev.is_screen_sharing:
                        line += " (screen share)"
                    if ev.ocr_text:
                        line += f" [Screen text: {ev.ocr_text[:100]}]"
                    ctx_parts.append(line)

    # 4. Visual context overview
    screen_share_count = job.get("screen_share_frames", 0)
    total_frames = job.get("visual_context_count", 0)
    if total_frames > 0:
        ctx_parts.append(
            f"\nVisual Analysis: {screen_share_count}/{total_frames} "
            f"frames had screen sharing detected"
        )

    # 5. Global Meeting Directory
    tm = get_temporal_memory()
    available_meetings = []
    # Deduplicate titles to keep the prompt clean if names repeat
    seen_titles = set()
    for m in tm.meetings.values():
        title = m.get('title', m.get('_id'))
        if title not in seen_titles:
            date_str = m.get('date', '')[:10]
            if date_str:
                available_meetings.append(f"- {title} ({date_str})")
            else:
                available_meetings.append(f"- {title}")
            seen_titles.add(title)
            
    if available_meetings:
        ctx_parts.append("\n--- GLOBAL KNOWLEDGE GRAPH DIRECTORY ---")
        ctx_parts.append("The following meetings are available in the system graph database. You can use the 'search_graph' tool to look up decisions or events from these past meetings:")
        ctx_parts.append("\n".join(available_meetings))

    # 6. Full transcript (cap to ~1000 words to fit model window of 4096 tokens)
    transcript = job.get("transcript", "")
    if transcript:
        words = transcript.split()
        if len(words) > 1000:
            transcript = " ".join(words[:1000]) + "\n... (transcript truncated to fit LLM window)"
        ctx_parts.append(f"\n--- FULL TRANSCRIPT ---\n{transcript}")

    meeting_context = "\n".join(ctx_parts)

    # ── Tool Definition: Search Graph ──────────────────────────────────────
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_graph",
                "description": (
                    "Search the cross-meeting knowledge graph for historical decisions, context, and events. "
                    "Use this if the user asks about past meetings or previous decisions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "Comma-separated keywords or entity names to search (e.g., 'cost, lcd screen, remote')."
                        }
                    },
                    "required": ["keywords"]
                }
            }
        }
    ]
    
    global mcp_tools_cache
    if mcp_tools_cache:
        tools.extend(mcp_tools_cache)

    main_loop = asyncio.get_running_loop()

    def tool_handler(fn_name: str, args: dict) -> str:
        if fn_name == "search_graph":
            keywords = args.get("keywords", "")
            if not keywords:
                return "Error: keywords not provided."
                
            tm_instance = get_temporal_memory()
            kws = [k.strip() for k in keywords.split(",") if k.strip()]
            
            all_items = []
            for kw in kws[:3]:
                items = tm_instance.query_temporal_context(kw, top_k=3)
                all_items.extend(items)
                
            seen = set()
            unique_items = []
            for item in all_items:
                uid = f"{item['meeting_id']}_{item['timestamp']}"
                if uid not in seen:
                    seen.add(uid)
                    unique_items.append(item)

            # ── Fallback: search by meeting title ─────────────────────
            # If entity search found nothing, try matching keywords
            # against meeting titles and return top events from matches.
            if not unique_items:
                for kw in kws[:3]:
                    kw_lower = kw.lower()
                    for mid, meta in tm_instance.meetings.items():
                        title = (meta.get("title") or "").lower()
                        if kw_lower in title:
                            event_ids = tm_instance.events_by_meeting.get(mid, [])
                            for eid in event_ids:
                                ev = tm_instance.events.get(eid)
                                if not ev:
                                    continue
                                ent = tm_instance.entities.get(ev.entity_id)
                                ent_name = ent.canonical_name if ent else "Unknown"
                                minutes = int(ev.start_time // 60)
                                seconds = int(ev.start_time % 60)
                                time_str = f"{minutes}:{seconds:02d}"
                                unique_items.append({
                                    'entity': ent_name,
                                    'entity_type': ent.type if ent else 'topic',
                                    'event_type': ev.event_type,
                                    'summary': ev.summary,
                                    'speaker': ev.speaker,
                                    'meeting_title': meta.get("title", mid[:8]),
                                    'meeting_id': mid,
                                    'timestamp': time_str,
                                    'sentiment': ev.sentiment,
                                    'unresolved_score': ent.unresolved_score if ent else 0,
                                    'relevance_score': ev.confidence,
                                    'citation': f"({meta.get('title', mid[:8])}, {ev.speaker}, {time_str})",
                                })
                            break  # found a matching meeting, stop

                # Sort by importance for meeting-title results
                importance_order = {'decision': 0, 'problem': 1, 'risk': 2, 'idea': 3, 'deadline': 4}
                unique_items.sort(key=lambda x: (importance_order.get(x['event_type'], 9), -x.get('relevance_score', 0)))
            else:
                unique_items.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            if not unique_items:
                return f"No historical events found for keywords: {keywords}"
                
            lines = []
            for item in unique_items[:10]:
                state = " ⚠️ UNRESOLVED" if item['unresolved_score'] > 0.6 else ""
                if item['event_type'] == 'decision': state = " ✅ DECIDED"
                lines.append(
                    f"- [{item['event_type'].upper()}] {item['entity']}: "
                    f"{item['summary'][:350]}{state} "
                    f"{item['citation']}"
                )
            return "\n".join(lines)
            
        # ── Fallback to MCP Tools ─────────────────────
        global mcp_session
        if mcp_session:
            try:
                # Execute the async MCP call on the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    mcp_session.call_tool(fn_name, arguments=args),
                    main_loop
                )
                result = future.result(timeout=30)
                if result.isError:
                    return f"MCP Tool Error: {result.content}"
                
                # Default text extraction from MCP result
                return "\n".join(c.text for c in result.content if getattr(c, 'type', '') == 'text')
            except Exception as e:
                logger.error(f"MCP Exception executing {fn_name}: {e}")
                return f"MCP Tool System Error: {e}"
        
        return f"Unknown tool: {fn_name}"

    try:
        from src.llm_summarizer import LLMSummarizer
        llm = LLMSummarizer()
        if not llm.is_ready:
            raise HTTPException(503, "LLM (LM Studio) is not available")
    except Exception as e:
        logger.error(f"LLM import failed: {e}")
        raise HTTPException(500, "Internal component error")

    system_prompt = f"""You are an intelligent AI assistant answering questions about a project.
You have access to the current meeting's data below.
To answer questions about PAST meetings or the complete history of decisions, YOU MUST CALL THE `search_graph` tool. 

--- CURRENT MEETING DATA ---
{meeting_context}

IMPORTANT INSTRUCTIONS:
1. Always base your answers on the transcripts and graph data.
2. Provide citations when stating facts (e.g., `(Meeting ES2002b, Speaker_A, 12:45)`).
3. Be concise."""

    # ── Build conversation history ─────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    
    raw_history = []
    for turn in history[-5:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        # role in OpenAI is 'user' or 'assistant'
        raw_history.append({"role": role if role in ['user', 'assistant'] else 'user', "content": content})

    raw_history.append({"role": "user", "content": user_msg})
    
    # Strictly merge consecutive messages of the same role to prevent MLX template crashes
    for msg in raw_history:
        if messages[-1]["role"] == msg["role"]:
            messages[-1]["content"] += f"\n\n{msg['content']}"
        else:
            messages.append(msg)

    # ── SSE Streaming Agent Loop ──────────────────────────────────────────
    import queue as _queue      # thread-safe queue for sync→async bridge

    step_queue = _queue.Queue()

    def _step_callback(step: dict):
        """Called from the sync agent thread — pushes events to the queue."""
        step_queue.put(step)

    async def _event_stream():
        """Async generator that yields SSE events."""
        loop = asyncio.get_event_loop()

        # Run agent_chat in a thread (it's synchronous / blocking)
        fut = loop.run_in_executor(
            None,
            lambda: llm.agent_chat(
                messages, tools, tool_handler,
                max_turns=5, step_callback=_step_callback,
            ),
        )

        # Poll the queue for intermediate steps while the agent runs
        while not fut.done():
            await asyncio.sleep(0.1)
            while not step_queue.empty():
                step = step_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'step', 'step': step})}\n\n"

        # Drain any remaining steps
        while not step_queue.empty():
            step = step_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'step', 'step': step})}\n\n"

        # Get the final result
        try:
            result = fut.result()
            yield f"data: {json.dumps({'type': 'reply', 'content': result['reply'].strip(), 'steps': result.get('steps', []), 'model': result.get('model', 'unknown')})}\n\n"
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    from starlette.responses import StreamingResponse as _SR
    return _SR(_event_stream(), media_type="text/event-stream")


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
        # Filter out filler/noise entities from graph visualization
        name_lower = ent.canonical_name.lower().strip()
        import re as _re
        from src.temporal_graph_memory import FILLER_SET, DISCARD_ENTITIES
        stripped = _re.sub(r'[^\w\s]', '', name_lower).strip()
        if stripped in FILLER_SET or name_lower in DISCARD_ENTITIES:
            continue
        words = stripped.split()
        if words and all(w in FILLER_SET for w in words):
            continue
        if len(stripped) < 3:
            continue

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
#  SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/system/reset")
def reset_system():
    """Clear all memory, jobs, embeddings, and uploaded videos."""
    global _jobs, temporal_memory, text_analyzer
    # 1. Clear jobs in memory
    _jobs.clear()
    _save_jobs()
    
    # 2. Reset temporal memory
    if temporal_memory is not None:
        temporal_memory.meetings.clear()
        temporal_memory.events.clear()
        temporal_memory.entities.clear()
        temporal_memory.events_by_meeting.clear()
        temporal_memory.events_by_entity.clear()
        temporal_memory.entities_by_type.clear()
        temporal_memory.save()
        
    # 3. Clear file system DATA_DIR (temporal_memory, jobs.json, participants.json)
    try:
        if MEMORY_DIR.exists():
            shutil.rmtree(MEMORY_DIR)
            MEMORY_DIR.mkdir()
    except Exception as e:
        logger.error(f"Error clearing MEMORY_DIR: {e}")

    try:
        if JOBS_FILE.exists():
            JOBS_FILE.unlink()
    except Exception as e:
        pass

    try:
        # We can keep roles or delete them, let's keep roles but delete jobs
        pass
    except Exception:
        pass

    # 4. Clear UPLOADS_DIR
    try:
        if UPLOADS_DIR.exists():
            for f in UPLOADS_DIR.iterdir():
                if f.is_file():
                    f.unlink()
    except Exception as e:
        logger.error(f"Error clearing UPLOADS_DIR: {e}")
        
    return {"status": "ok", "message": "Memory reset successfully"}

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
