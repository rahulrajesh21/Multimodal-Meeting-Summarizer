"""
Teams Media Server
==================
Mimics the Microsoft Graph / Teams transcript & recording REST API.

Endpoints (Teams Graph API–style):
  GET  /v1.0/me/onlineMeetings                              – list meetings
  POST /v1.0/me/onlineMeetings                              – create meeting
  GET  /v1.0/me/onlineMeetings/{mid}                        – get meeting
  DELETE /v1.0/me/onlineMeetings/{mid}                      – delete meeting

  GET  /v1.0/me/onlineMeetings/{mid}/transcripts            – list transcripts
  POST /v1.0/me/onlineMeetings/{mid}/transcripts            – upload transcript
  GET  /v1.0/me/onlineMeetings/{mid}/transcripts/{tid}      – get transcript meta
  GET  /v1.0/me/onlineMeetings/{mid}/transcripts/{tid}/content – raw VTT content
  DELETE /v1.0/me/onlineMeetings/{mid}/transcripts/{tid}    – delete transcript

  GET  /v1.0/me/onlineMeetings/{mid}/recordings             – list recordings
  POST /v1.0/me/onlineMeetings/{mid}/recordings             – upload recording
  GET  /v1.0/me/onlineMeetings/{mid}/recordings/{rid}       – get recording meta
  GET  /v1.0/me/onlineMeetings/{mid}/recordings/{rid}/content – stream video
  DELETE /v1.0/me/onlineMeetings/{mid}/recordings/{rid}     – delete recording

  GET  /                                                     – Admin UI
  GET  /health                                               – health check

Run:  uvicorn teams_media_server.server:app --port 8001 --reload
  or: python -m teams_media_server.server
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── storage layout ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
STORAGE_DIR  = BASE_DIR / "storage"
META_FILE    = STORAGE_DIR / "meetings.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── load / save helpers ───────────────────────────────────────────────────────
def _load() -> dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except Exception:
            pass
    return {"meetings": {}}


def _save(db: dict) -> None:
    META_FILE.write_text(json.dumps(db, indent=2))


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Teams Media Server",
    description="Microsoft Teams–compatible transcript & recording API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── timestamp helper ──────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ═══════════════════════════════════════════════════════════════════════════════
#  MEETINGS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1.0/me/onlineMeetings")
def list_meetings():
    db = _load()
    meetings = list(db["meetings"].values())
    return {"@odata.context": "https://graph.microsoft.com/v1.0/$metadata#users/onlineMeetings",
            "@odata.count": len(meetings),
            "value": meetings}


@app.post("/v1.0/me/onlineMeetings", status_code=201)
async def create_meeting(
    subject: str = Form(...),
    organizer: str = Form(""),
    start_datetime: str = Form(""),
    end_datetime: str = Form(""),
):
    mid = f"MSSpkl~{uuid.uuid4().hex[:24]}"
    now = _now()
    meeting = {
        "id": mid,
        "subject": subject,
        "organizer": {"displayName": organizer or "Unknown Organizer"},
        "startDateTime": start_datetime or now,
        "endDateTime": end_datetime or now,
        "createdDateTime": now,
        "joinUrl": f"https://teams.microsoft.com/l/meetup-join/{mid}",
        "transcripts": [],
        "recordings": [],
    }
    db = _load()
    db["meetings"][mid] = meeting
    _save(db)
    (STORAGE_DIR / mid).mkdir(parents=True, exist_ok=True)
    return meeting


@app.get("/v1.0/me/onlineMeetings/{mid}")
def get_meeting(mid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, f"Meeting {mid!r} not found")
    return m


@app.delete("/v1.0/me/onlineMeetings/{mid}", status_code=204)
def delete_meeting(mid: str):
    db = _load()
    if mid not in db["meetings"]:
        raise HTTPException(404, f"Meeting {mid!r} not found")
    del db["meetings"][mid]
    _save(db)
    shutil.rmtree(STORAGE_DIR / mid, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1.0/me/onlineMeetings/{mid}/transcripts")
def list_transcripts(mid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    return {"@odata.context": "https://graph.microsoft.com/v1.0/$metadata#users/onlineMeetings/transcripts",
            "value": m.get("transcripts", [])}


@app.post("/v1.0/me/onlineMeetings/{mid}/transcripts", status_code=201)
async def upload_transcript(
    mid: str,
    file: UploadFile = File(...),
    content_correlation_id: str = Form(""),
):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")

    tid        = f"VttTranscript_{uuid.uuid4().hex[:16]}"
    fname      = f"{tid}_{file.filename or 'transcript.vtt'}"
    dest_dir   = STORAGE_DIR / mid / "transcripts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest       = dest_dir / fname

    async with aiofiles.open(dest, "wb") as f_out:
        while chunk := await file.read(1 << 20):
            await f_out.write(chunk)

    # Parse speaker count from VTT (quick heuristic)
    vtt_text  = dest.read_text(errors="replace")
    
    if "WEBVTT" not in vtt_text[:100]:
        vtt_text = _convert_txt_to_vtt(vtt_text)
        dest.write_text(vtt_text)
        
    speakers  = _parse_vtt_speakers(vtt_text)
    segments  = _parse_vtt_segments(vtt_text)

    meta = {
        "id": tid,
        "meetingId": mid,
        "createdDateTime": _now(),
        "transcriptContentUrl": f"/v1.0/me/onlineMeetings/{mid}/transcripts/{tid}/content",
        "contentCorrelationId": content_correlation_id or tid,
        "filename": file.filename or "transcript.vtt",
        "storedFilename": fname,
        "speakerCount": len(speakers),
        "speakers": speakers,
        "segmentCount": len(segments),
        "diarization": [
            {
                "speakerLabel": s,
                "displayName": s,
                "segments": [seg for seg in segments if seg["speaker"] == s],
            }
            for s in speakers
        ],
    }

    m.setdefault("transcripts", []).append(meta)
    _save(db)
    return meta


@app.get("/v1.0/me/onlineMeetings/{mid}/transcripts/{tid}")
def get_transcript_meta(mid: str, tid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    for t in m.get("transcripts", []):
        if t["id"] == tid:
            return t
    raise HTTPException(404, "Transcript not found")


@app.get("/v1.0/me/onlineMeetings/{mid}/transcripts/{tid}/content")
def get_transcript_content(mid: str, tid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    for t in m.get("transcripts", []):
        if t["id"] == tid:
            path = STORAGE_DIR / mid / "transcripts" / t["storedFilename"]
            if not path.exists():
                raise HTTPException(404, "Transcript file not found on disk")
            return FileResponse(str(path), media_type="text/vtt",
                                filename=t.get("filename", "transcript.vtt"))
    raise HTTPException(404, "Transcript not found")


@app.delete("/v1.0/me/onlineMeetings/{mid}/transcripts/{tid}", status_code=204)
def delete_transcript(mid: str, tid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    txs = m.get("transcripts", [])
    for i, t in enumerate(txs):
        if t["id"] == tid:
            (STORAGE_DIR / mid / "transcripts" / t["storedFilename"]).unlink(missing_ok=True)
            txs.pop(i)
            _save(db)
            return
    raise HTTPException(404, "Transcript not found")


# ═══════════════════════════════════════════════════════════════════════════════
#  RECORDINGS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1.0/me/onlineMeetings/{mid}/recordings")
def list_recordings(mid: str):
    db = _load()
    m = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    return {"@odata.context": "https://graph.microsoft.com/v1.0/$metadata#users/onlineMeetings/recordings",
            "value": m.get("recordings", [])}


@app.post("/v1.0/me/onlineMeetings/{mid}/recordings", status_code=201)
async def upload_recording(
    mid: str,
    file: UploadFile = File(...),
):
    db = _load()
    m  = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")

    rid      = f"Recording_{uuid.uuid4().hex[:16]}"
    fname    = f"{rid}_{file.filename or 'recording.mp4'}"
    dest_dir = STORAGE_DIR / mid / "recordings"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest     = dest_dir / fname

    async with aiofiles.open(dest, "wb") as f_out:
        while chunk := await file.read(1 << 20):
            await f_out.write(chunk)

    size = dest.stat().st_size

    meta = {
        "id": rid,
        "meetingId": mid,
        "createdDateTime": _now(),
        "recordingContentUrl": f"/v1.0/me/onlineMeetings/{mid}/recordings/{rid}/content",
        "filename": file.filename or "recording.mp4",
        "storedFilename": fname,
        "fileSizeBytes": size,
        "duration": None,
    }

    m.setdefault("recordings", []).append(meta)
    _save(db)
    return meta


@app.get("/v1.0/me/onlineMeetings/{mid}/recordings/{rid}")
def get_recording_meta(mid: str, rid: str):
    db = _load()
    m  = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    for r in m.get("recordings", []):
        if r["id"] == rid:
            return r
    raise HTTPException(404, "Recording not found")


@app.get("/v1.0/me/onlineMeetings/{mid}/recordings/{rid}/content")
def stream_recording(mid: str, rid: str):
    db = _load()
    m  = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    for r in m.get("recordings", []):
        if r["id"] == rid:
            path = STORAGE_DIR / mid / "recordings" / r["storedFilename"]
            if not path.exists():
                raise HTTPException(404, "Recording file not found on disk")
            mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
            return FileResponse(str(path), media_type=mime,
                                filename=r.get("filename", "recording.mp4"))
    raise HTTPException(404, "Recording not found")


@app.delete("/v1.0/me/onlineMeetings/{mid}/recordings/{rid}", status_code=204)
def delete_recording(mid: str, rid: str):
    db = _load()
    m  = db["meetings"].get(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    recs = m.get("recordings", [])
    for i, r in enumerate(recs):
        if r["id"] == rid:
            (STORAGE_DIR / mid / "recordings" / r["storedFilename"]).unlink(missing_ok=True)
            recs.pop(i)
            _save(db)
            return
    raise HTTPException(404, "Recording not found")


# ═══════════════════════════════════════════════════════════════════════════════
#  VTT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_txt_to_vtt(text: str) -> str:
    """Converts a Teams TXT transcript to WebVTT format."""
    import re
    if "WEBVTT" in text[:100]:
        return text
    
    parts = re.split(r"(\[\d{2}:\d{2}:\d{2}\]\s+[^:]+:)", text)
    segments = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s+([^:]+):", header)
        if m:
            segments.append({
                "time": m.group(1),
                "speaker": m.group(2).strip(),
                "text": content
            })
            
    if not segments:
        return text
        
    vtt = ["WEBVTT", ""]
    for i, seg in enumerate(segments):
        start = f"{seg['time']}.000"
        if len(start) == 9:
            start = f"00:{start}"
            
        if i + 1 < len(segments):
            end = f"{segments[i+1]['time']}.000"
            if len(end) == 9:
                end = f"00:{end}"
        else:
            parts_time = start.split(':')
            h, m, s = int(parts_time[0]), int(parts_time[1]), float(parts_time[2])
            s += 10.0
            if s >= 60:
                s -= 60
                m += 1
            if m >= 60:
                m -= 60
                h += 1
            end = f"{h:02d}:{m:02d}:{s:06.3f}"
            
        vtt.append(f"{start} --> {end}")
        vtt.append(f"<v {seg['speaker']}>{seg['text']}")
        vtt.append("")
        
    return "\n".join(vtt)


def _parse_vtt_speakers(vtt: str) -> list[str]:
    """Extract unique speaker labels from a VTT file."""
    import re
    labels: list[str] = []
    seen: set[str]    = set()
    # Match  <v Speaker Name>   or  Speaker Name:  at the start of a cue
    for m in re.finditer(r"<v\s+([^>]+)>", vtt):
        label = m.group(1).strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _parse_vtt_segments(vtt: str) -> list[dict]:
    """Parse VTT cues into structured segments."""
    import re
    segments: list[dict] = []
    blocks = re.split(r"\n{2,}", vtt.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        # Find timestamp line
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue
        parts       = ts_line.split("-->")
        start_raw   = parts[0].strip()
        end_raw     = parts[1].strip().split()[0]
        text_lines  = [l for l in lines if "-->" not in l and l != "WEBVTT"
                       and not l.isdigit() and not re.match(r"^\d{2}:\d{2}", l)]
        full_text   = " ".join(text_lines)

        # Extract speaker from <v Speaker>text</v>
        speaker_m   = re.match(r"<v\s+([^>]+)>(.*)", full_text)
        if speaker_m:
            speaker = speaker_m.group(1).strip()
            text    = re.sub(r"<[^>]+>", "", speaker_m.group(2)).strip()
        else:
            speaker = "Unknown"
            text    = re.sub(r"<[^>]+>", "", full_text).strip()

        segments.append({
            "speaker": speaker,
            "start":   start_raw,
            "end":     end_raw,
            "text":    text,
        })
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    db = _load()
    return {"status": "ok", "meetings": len(db["meetings"]), "time": _now()}


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN UI  (served at GET /)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def admin_ui():
    """Serve the Teams Media Server admin UI."""
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Teams Media Server</h1><p>Template not found.</p>", status_code=500)


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("teams_media_server.server:app", host="0.0.0.0", port=8001, reload=True)
