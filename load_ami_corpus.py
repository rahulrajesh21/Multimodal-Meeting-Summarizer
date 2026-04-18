"""
AMI Meeting Corpus → TemporalGraphMemory Ingestion Script

Parses AMI NXT-format word XML files and feeds diarized transcript segments
into the cross-meeting graph for testing entity resolution across meetings.

Usage:
    python load_ami_corpus.py [--corpus-dir /path/to/ami_public_manual_1.6.2/words]
                              [--meetings ES2002a ES2002b ES2002c ES2002d]
                              [--storage-path meeting_memory]

The script:
  1. Parses word-level XML files per speaker (e.g., ES2002a.A.words.xml)
  2. Merges words into sentence-level segments using pause detection (>1s gap)
  3. Interleaves all speakers chronologically
  4. Feeds each segment into TemporalGraphMemory.add_segment()
  5. Prints a summary of entities and cross-meeting connections found
"""

import os
import sys
import glob
import argparse
import html
from xml.etree import ElementTree as ET
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.temporal_graph_memory import TemporalGraphMemory


# ──────────────────────────────────────────────────────────────────────────────
# AMI XML Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_ami_words_xml(filepath: str) -> list:
    """
    Parse a single AMI words XML file (e.g., ES2002a.A.words.xml).
    
    Returns a list of word dicts:
      [{"word": "Hi", "start": 77.44, "end": 77.74}, ...]
    """
    ns = {"nite": "http://nite.sourceforge.net/"}
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    words = []
    for w_elem in root.findall("w", ns):
        start = w_elem.get("starttime")
        end = w_elem.get("endtime")
        is_punc = w_elem.get("punc") == "true"
        text = w_elem.text or ""
        
        # Decode HTML entities (&#39; → ')
        text = html.unescape(text)
        
        if start is not None and end is not None and text.strip():
            words.append({
                "word": text.strip(),
                "start": float(start),
                "end": float(end),
                "is_punc": is_punc,
            })
    
    return words


def words_to_sentences(words: list, pause_threshold: float = 1.5) -> list:
    """
    Merge word-level data into sentence-level segments.
    
    A new sentence starts when:
      - There's a pause > pause_threshold seconds between consecutive words
      - OR current sentence exceeds 30 words
    
    Returns: [{"text": "...", "start": 77.44, "end": 80.87}, ...]
    """
    if not words:
        return []
    
    sentences = []
    current_words = [words[0]]
    
    for i in range(1, len(words)):
        prev_end = words[i - 1]["end"]
        curr_start = words[i]["start"]
        gap = curr_start - prev_end
        
        # Split on long pauses or sentence length
        if gap > pause_threshold or len(current_words) > 30:
            # Build the sentence text
            text = _build_sentence_text(current_words)
            if text.strip():
                sentences.append({
                    "text": text.strip(),
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                })
            current_words = [words[i]]
        else:
            current_words.append(words[i])
    
    # Flush remaining
    if current_words:
        text = _build_sentence_text(current_words)
        if text.strip():
            sentences.append({
                "text": text.strip(),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
            })
    
    return sentences


def _build_sentence_text(words: list) -> str:
    """Join words, attaching punctuation directly to the preceding word."""
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


def load_meeting(corpus_dir: str, meeting_id: str) -> list:
    """
    Load all speaker files for a meeting and produce interleaved segments.
    
    Files follow the pattern: {meeting_id}.{A|B|C|D}.words.xml
    
    Returns: [{"text": "...", "start": 77.44, "end": 80.87, "speaker": "Speaker_A"}, ...]
             sorted chronologically.
    """
    pattern = os.path.join(corpus_dir, f"{meeting_id}.*.words.xml")
    speaker_files = sorted(glob.glob(pattern))
    
    if not speaker_files:
        print(f"  ⚠ No files found for {meeting_id} in {corpus_dir}")
        return []
    
    all_segments = []
    speaker_names = {}  # Map A/B/C/D to names found in text
    
    for fpath in speaker_files:
        # Extract speaker letter from filename (e.g., "A" from "ES2002a.A.words.xml")
        basename = os.path.basename(fpath)
        parts = basename.split(".")
        speaker_letter = parts[1]  # A, B, C, or D
        speaker_id = f"Speaker_{speaker_letter}"
        
        words = parse_ami_words_xml(fpath)
        sentences = words_to_sentences(words)
        
        for seg in sentences:
            seg["speaker"] = speaker_id
        
        all_segments.extend(sentences)
        print(f"  📄 {basename}: {len(words)} words → {len(sentences)} segments")
    
    # Sort all segments chronologically across speakers
    all_segments.sort(key=lambda s: s["start"])
    
    return all_segments


# ──────────────────────────────────────────────────────────────────────────────
# Graph Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest_meeting(graph: TemporalGraphMemory, meeting_id: str, segments: list):
    """Feed parsed segments into the TemporalGraphMemory with sliding-window context."""
    
    participants = list(set(s["speaker"] for s in segments))
    
    # Create meeting node
    gid = graph.create_meeting(
        title=f"AMI Meeting {meeting_id}",
        participants=participants,
        tags=["ami-corpus", "product-design", "remote-control"],
    )
    
    print(f"\n  🔄 Ingesting {len(segments)} segments into graph (with sliding window)...")
    
    for i, seg in enumerate(segments):
        # Build sliding window context
        prev_text = segments[i - 1]["text"] if i > 0 else None
        next_text = segments[i + 1]["text"] if i < len(segments) - 1 else None
        
        graph.add_segment(
            meeting_id=gid,
            text=seg["text"],
            start_time=seg["start"],
            end_time=seg["end"],
            speaker=seg["speaker"],
            prev_text=prev_text,
            next_text=next_text,
        )
        
        # Progress indication every 20 segments
        if (i + 1) % 20 == 0:
            print(f"    ✓ {i + 1}/{len(segments)} segments processed")
    
    print(f"  ✅ Meeting {meeting_id} ingested")
    return gid


def print_graph_summary(graph: TemporalGraphMemory):
    """Print a summary of the graph state after ingestion."""
    stats = graph.get_statistics()
    
    print("\n" + "=" * 60)
    print("📊 CROSS-MEETING GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Meetings:   {stats.get('meetings', 0)}")
    print(f"  Entities:   {stats.get('entities', 0)}")
    print(f"  Events:     {stats.get('events', 0)}")
    
    # Show entities by type
    if graph.entities_by_type:
        print(f"\n  Entities by type:")
        for etype, eids in graph.entities_by_type.items():
            print(f"    {etype}: {len(eids)}")
    
    # Show cross-meeting entities (entities mentioned in >1 meeting)
    cross_meeting = []
    for eid, entity in graph.entities.items():
        meeting_ids = set()
        for ev_id in graph.events_by_entity.get(eid, []):
            ev = graph.events.get(ev_id)
            if ev:
                meeting_ids.add(ev.meeting_id)
        if len(meeting_ids) > 1:
            cross_meeting.append((entity.canonical_name, entity.type, len(meeting_ids), entity.mention_count))
    
    if cross_meeting:
        cross_meeting.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  🔗 CROSS-MEETING ENTITIES ({len(cross_meeting)} found):")
        print(f"  {'Entity':<30} {'Type':<15} {'Meetings':<10} {'Mentions':<10}")
        print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*10}")
        for name, etype, n_meetings, mentions in cross_meeting[:20]:
            print(f"  {name:<30} {etype:<15} {n_meetings:<10} {mentions:<10}")
    else:
        print("\n  ⚠ No cross-meeting entities found yet.")
        print("    (This means no entities were similar enough across meetings)")
    
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load AMI corpus into TemporalGraphMemory")
    parser.add_argument(
        "--corpus-dir",
        default=os.path.expanduser("~/Downloads/ami_public_manual_1.6.2/words"),
        help="Path to AMI words XML directory",
    )
    parser.add_argument(
        "--meetings",
        nargs="+",
        default=["ES2002a", "ES2002b", "ES2002c", "ES2002d"],
        help="Meeting IDs to process (default: ES2002a-d)",
    )
    parser.add_argument(
        "--storage-path",
        default="data/temporal_memory",
        help="Graph storage directory (must match api/main.py MEMORY_DIR)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear existing graph data before ingesting (recommended for clean runs)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙 AMI Meeting Corpus → Cross-Meeting Graph Loader")
    print("=" * 60)
    print(f"  Corpus dir:   {args.corpus_dir}")
    print(f"  Meetings:     {', '.join(args.meetings)}")
    print(f"  Storage:      {args.storage_path}")
    print()
    
    # Check corpus exists
    if not os.path.isdir(args.corpus_dir):
        print(f"❌ Corpus directory not found: {args.corpus_dir}")
        sys.exit(1)
    
    # Initialize text analyzer for embeddings (CRITICAL for entity resolution)
    from src.text_analysis import TextAnalyzer
    print("  🔧 Loading TextAnalyzer for entity embeddings...")
    ta = TextAnalyzer()
    
    # Initialize graph WITH text_analyzer so embeddings are generated
    graph = TemporalGraphMemory(storage_path=args.storage_path, text_analyzer=ta)
    
    # Reset if requested (clear old duplicate data)
    if args.reset:
        print("  🗑️  Resetting graph data...")
        graph.meetings.clear()
        graph.events.clear()
        graph.entities.clear()
        graph.events_by_meeting.clear()
        graph.events_by_entity.clear()
        graph.entities_by_type.clear()
        graph._save_graph()
        print("  ✅ Graph cleared.")
    
    # Process each meeting
    for meeting_id in args.meetings:
        print(f"\n{'─' * 40}")
        print(f"📋 Loading meeting: {meeting_id}")
        print(f"{'─' * 40}")
        
        segments = load_meeting(args.corpus_dir, meeting_id)
        if segments:
            ingest_meeting(graph, meeting_id, segments)
        else:
            print(f"  ⚠ Skipping {meeting_id} — no segments")
    
    # Print summary
    print_graph_summary(graph)
    
    print(f"\n✅ Done! Graph saved to: {args.storage_path}/graph_db.json")
    print(f"   You can now use this graph in the Streamlit app or inspect it directly.")


if __name__ == "__main__":
    main()
