"""
AMI Meeting Corpus → Teams Media Server Uploader

Parses AMI NXT word XML files, converts to VTT format with speaker diarization,
and uploads each meeting to the Teams Media Server. After upload, you can process
the meetings through the normal Next.js UI pipeline.

Usage:
    python upload_ami_to_teams.py [--meetings ES2002a ES2002b ES2002c ES2002d]
                                  [--teams-url http://localhost:8001]
"""

import os
import sys
import glob
import argparse
import html
import io
import requests
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# AMI XML Parser (reused from load_ami_corpus.py)
# ──────────────────────────────────────────────────────────────────────────────

def parse_ami_words_xml(filepath: str) -> list:
    ns = {"nite": "http://nite.sourceforge.net/"}
    tree = ET.parse(filepath)
    root = tree.getroot()
    words = []
    for w_elem in root.findall("w", ns):
        start = w_elem.get("starttime")
        end = w_elem.get("endtime")
        is_punc = w_elem.get("punc") == "true"
        text = html.unescape(w_elem.text or "")
        if start is not None and end is not None and text.strip():
            words.append({
                "word": text.strip(),
                "start": float(start),
                "end": float(end),
                "is_punc": is_punc,
            })
    return words


def words_to_sentences(words, pause_threshold=1.5):
    if not words:
        return []
    sentences = []
    current = [words[0]]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > pause_threshold or len(current) > 30:
            text = _build_text(current)
            if text.strip():
                sentences.append({"text": text.strip(), "start": current[0]["start"], "end": current[-1]["end"]})
            current = [words[i]]
        else:
            current.append(words[i])
    if current:
        text = _build_text(current)
        if text.strip():
            sentences.append({"text": text.strip(), "start": current[0]["start"], "end": current[-1]["end"]})
    return sentences


def _build_text(words):
    parts = []
    for w in words:
        if w["is_punc"]:
            if parts:
                parts[-1] += w["word"]
            else:
                parts.append(w["word"])
        else:
            parts.append(w["word"])
    return " ".join(parts)


def load_meeting_segments(corpus_dir: str, meeting_id: str) -> list:
    """Load all speakers for a meeting and return chronological diarized segments."""
    pattern = os.path.join(corpus_dir, f"{meeting_id}.*.words.xml")
    speaker_files = sorted(glob.glob(pattern))
    if not speaker_files:
        print(f"  ⚠ No files found for {meeting_id}")
        return []

    all_segments = []
    for fpath in speaker_files:
        basename = os.path.basename(fpath)
        speaker_letter = basename.split(".")[1]
        speaker_id = f"Speaker_{speaker_letter}"

        words = parse_ami_words_xml(fpath)
        sentences = words_to_sentences(words)
        for seg in sentences:
            seg["speaker"] = speaker_id
        all_segments.extend(sentences)
        print(f"  📄 {basename}: {len(words)} words → {len(sentences)} segments")

    all_segments.sort(key=lambda s: s["start"])
    return all_segments


# ──────────────────────────────────────────────────────────────────────────────
# VTT Converter
# ──────────────────────────────────────────────────────────────────────────────

def seconds_to_vtt_ts(seconds: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def segments_to_vtt(segments: list) -> str:
    """Convert diarized segments to WebVTT format."""
    lines = ["WEBVTT", ""]

    for i, seg in enumerate(segments):
        start_ts = seconds_to_vtt_ts(seg["start"])
        end_ts = seconds_to_vtt_ts(seg["end"])
        speaker = seg["speaker"]
        text = seg["text"]

        lines.append(str(i + 1))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"<v {speaker}>{text}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Teams Server Upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_teams(teams_url: str, meeting_id: str, segments: list) -> dict:
    """Create a meeting on Teams server and upload VTT transcript."""

    # 1. Create meeting
    print(f"  📤 Creating meeting on Teams server...")
    r = requests.post(
        f"{teams_url}/v1.0/me/onlineMeetings",
        data={
            "subject": f"AMI Meeting {meeting_id}",
            "organizer": "AMI Corpus",
        },
    )
    r.raise_for_status()
    meeting = r.json()
    mid = meeting["id"]
    print(f"  ✅ Meeting created: {mid[:20]}...")

    # 2. Generate VTT
    vtt_content = segments_to_vtt(segments)
    vtt_bytes = vtt_content.encode("utf-8")

    # 3. Upload transcript
    print(f"  📤 Uploading VTT transcript ({len(segments)} segments, {len(vtt_bytes)} bytes)...")
    r = requests.post(
        f"{teams_url}/v1.0/me/onlineMeetings/{mid}/transcripts",
        files={"file": (f"{meeting_id}_transcript.vtt", io.BytesIO(vtt_bytes), "text/vtt")},
        data={"content_correlation_id": meeting_id},
    )
    r.raise_for_status()
    transcript = r.json()
    print(f"  ✅ Transcript uploaded: {transcript['id'][:20]}...")
    print(f"     Speakers: {transcript.get('speakerCount', '?')}, Segments: {transcript.get('segmentCount', '?')}")

    return {
        "meeting_id": mid,
        "transcript_id": transcript["id"],
        "subject": meeting.get("subject", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload AMI corpus to Teams Media Server")
    parser.add_argument(
        "--corpus-dir",
        default=os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/words"),
        help="Path to AMI words XML directory",
    )
    parser.add_argument(
        "--meetings", nargs="+",
        default=["ES2002a", "ES2002b", "ES2002c", "ES2002d"],
        help="Meeting IDs to upload",
    )
    parser.add_argument(
        "--teams-url",
        default="http://localhost:8001",
        help="Teams Media Server URL",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🎙 AMI Meeting Corpus → Teams Media Server Uploader")
    print("=" * 60)
    print(f"  Corpus dir:   {args.corpus_dir}")
    print(f"  Meetings:     {', '.join(args.meetings)}")
    print(f"  Teams server: {args.teams_url}")
    print()

    if not os.path.isdir(args.corpus_dir):
        print(f"❌ Corpus directory not found: {args.corpus_dir}")
        sys.exit(1)

    # Check Teams server is running
    try:
        r = requests.get(f"{args.teams_url}/health", timeout=3)
        r.raise_for_status()
        print("  ✅ Teams server is running\n")
    except Exception as e:
        print(f"  ❌ Teams server not reachable at {args.teams_url}: {e}")
        print(f"     Start it with: cd teams_media_server && python -m uvicorn server:app --port 8001")
        sys.exit(1)

    uploaded = []
    for meeting_id in args.meetings:
        print(f"\n{'─' * 40}")
        print(f"📋 Processing: {meeting_id}")
        print(f"{'─' * 40}")

        segments = load_meeting_segments(args.corpus_dir, meeting_id)
        if not segments:
            print(f"  ⚠ Skipping {meeting_id} — no segments")
            continue

        result = upload_to_teams(args.teams_url, meeting_id, segments)
        uploaded.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 UPLOAD SUMMARY")
    print("=" * 60)
    for u in uploaded:
        print(f"  ✅ {u['subject']}")
        print(f"     Meeting:    {u['meeting_id'][:20]}...")
        print(f"     Transcript: {u['transcript_id'][:20]}...")
    print(f"\n  Total: {len(uploaded)} meetings uploaded")
    print(f"\n  👉 Now open http://localhost:3000 and process these meetings from the UI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
