"""
Unit tests for SpeakerIdentifier and ParticipantStore speaker mapping.
Run from the project root:  python tests/test_speaker_identifier.py
"""

import os
import sys
import shutil
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.speaker_identifier import SpeakerIdentifier, LOW_CONFIDENCE_THRESHOLD
from src.participant_store import ParticipantStore

TEST_DIR = "test_si_data"


def check(condition, label):
    status = "✅" if condition else "❌"
    print(f"  {status} {label}")
    return condition


# ─── Helper ───────────────────────────────────────────────────────────────────

def make_fake_segments(speakers=("SPEAKER_00", "SPEAKER_01"), texts=None):
    """Build minimal diarization segment list."""
    if texts is None:
        texts = ["Hello team.", "Thanks Alice."]
    segs = []
    t = 0.0
    for i, (spk, txt) in enumerate(zip(speakers, texts)):
        segs.append({"start": t, "end": t + 3.0, "speaker": spk, "text": txt})
        t += 3.0
    return segs


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_order_heuristic_mapping():
    print("\n── Order Heuristic Mapping ──")
    all_pass = True
    si = SpeakerIdentifier()

    speaker_ids = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    names = ["Alice", "Bob", "Carol"]

    result = si._order_heuristic_mapping(speaker_ids, names)

    all_pass &= check(
        result["SPEAKER_00"][0] == "Alice",
        "SPEAKER_00 mapped to first registered name (Alice)"
    )
    all_pass &= check(
        result["SPEAKER_01"][0] == "Bob",
        "SPEAKER_01 mapped to second registered name (Bob)"
    )
    all_pass &= check(
        result["SPEAKER_02"][0] == "Carol",
        "SPEAKER_02 mapped to third registered name (Carol)"
    )
    all_pass &= check(
        0.0 < result["SPEAKER_00"][1] <= 0.60,
        f"SPEAKER_00 confidence is in expected range (got {result['SPEAKER_00'][1]:.3f})"
    )
    # Extra speaker with no matched name
    result_extra = si._order_heuristic_mapping(["SPEAKER_00", "SPEAKER_01"], ["Alice"])
    all_pass &= check(
        result_extra["SPEAKER_01"] == (None, 0.0),
        "Unmatched speaker returns (None, 0.0)"
    )
    return all_pass


def test_transcript_name_mapping():
    print("\n── Transcript Name Mapping ──")
    all_pass = True
    si = SpeakerIdentifier()

    # Alice talks about budgets; Bob is called out by Alice
    speaker_text_map = {
        "SPEAKER_00": "we need to finalise the budget Alice mentioned last time",
        "SPEAKER_01": "Carol should review the designs Carol highlighted previously",
    }
    registered = ["Alice Johnson", "Bob Smith", "Carol Lee"]

    result = si._transcript_name_mapping(speaker_text_map, registered)

    all_pass &= check(
        result["SPEAKER_00"][0] == "Alice Johnson",
        f"SPEAKER_00: detected 'Alice' mention → Alice Johnson (got {result['SPEAKER_00'][0]!r})"
    )
    all_pass &= check(
        result["SPEAKER_00"][1] >= 0.50,
        f"SPEAKER_00: confidence ≥ 0.50 (got {result['SPEAKER_00'][1]:.3f})"
    )
    all_pass &= check(
        result["SPEAKER_01"][0] == "Carol Lee",
        f"SPEAKER_01: detected 'Carol' mention → Carol Lee (got {result['SPEAKER_01'][0]!r})"
    )

    # Edge case: no mentions
    result_empty = si._transcript_name_mapping({"SPEAKER_02": "blah blah xyz"}, registered)
    all_pass &= check(
        result_empty["SPEAKER_02"] == (None, 0.0),
        "No name mention → (None, 0.0)"
    )
    return all_pass


def test_fusion_prefers_higher_confidence():
    print("\n── Fusion Signal Preference ──")
    all_pass = True
    si = SpeakerIdentifier()

    speaker_ids = ["SPEAKER_00"]
    # Order says 'Alice' with low confidence
    order_map = {"SPEAKER_00": ("Alice", 0.30)}
    # Transcript says 'Bob' with high confidence
    transcript_map = {"SPEAKER_00": ("Bob", 0.80)}
    # No voice map
    voice_map = {}

    result = si._fuse_signals(speaker_ids, order_map, transcript_map, voice_map)
    all_pass &= check(
        result["SPEAKER_00"][0] == "Bob",
        f"Fusion picks high-confidence transcript signal 'Bob' over low-confidence order 'Alice' (got {result['SPEAKER_00'][0]!r})"
    )

    # Both signals agree → confidence should be higher than individual
    order_map2 = {"SPEAKER_00": ("Alice", 0.50)}
    transcript_map2 = {"SPEAKER_00": ("Alice", 0.75)}
    result2 = si._fuse_signals(speaker_ids, order_map2, transcript_map2, {})
    all_pass &= check(
        result2["SPEAKER_00"][0] == "Alice",
        "When signals agree, the same name is returned"
    )
    return all_pass


def test_build_mapping_end_to_end():
    print("\n── build_mapping() end-to-end ──")
    all_pass = True
    si = SpeakerIdentifier()

    segs = [
        {"start": 0.0,  "end": 3.0,  "speaker": "SPEAKER_00", "text": "Hello, this is Alice speaking."},
        {"start": 3.0,  "end": 6.0,  "speaker": "SPEAKER_01", "text": "Bob here. Alice made a good point."},
        {"start": 6.0,  "end": 9.0,  "speaker": "SPEAKER_00", "text": "Thanks Bob, I agree."},
    ]
    participants = ["Alice Smith", "Bob Chen"]
    dummy_audio = np.zeros(16000 * 10, dtype=np.float32)

    mapping = si.build_mapping(segs, dummy_audio, 16000, participants)

    all_pass &= check("SPEAKER_00" in mapping, "SPEAKER_00 is in the mapping")
    all_pass &= check("SPEAKER_01" in mapping, "SPEAKER_01 is in the mapping")
    all_pass &= check(
        mapping["SPEAKER_00"][0] in ("Alice Smith", "Bob Chen"),
        f"SPEAKER_00 resolved to a participant name (got {mapping['SPEAKER_00'][0]!r})"
    )
    # Alice should be detected by transcript name scan in SPEAKER_00's text
    all_pass &= check(
        mapping["SPEAKER_00"][0] == "Alice Smith",
        f"SPEAKER_00 correctly resolved to Alice Smith via transcript scan (got {mapping['SPEAKER_00'][0]!r})"
    )
    return all_pass


def test_save_load_speaker_mapping():
    print("\n── ParticipantStore Speaker Mapping Persistence ──")
    all_pass = True

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    store = ParticipantStore(data_dir=TEST_DIR)

    mapping = {
        "SPEAKER_00": ("Alice Johnson", 0.91),
        "SPEAKER_01": ("Bob Smith", 0.55),
        "SPEAKER_02": (None, 0.0),
    }
    store.save_speaker_mapping(mapping)

    # Check get_name_for_speaker()
    name, conf = store.get_name_for_speaker("SPEAKER_00")
    all_pass &= check(name == "Alice Johnson", f"get_name_for_speaker('SPEAKER_00') → 'Alice Johnson' (got {name!r})")
    all_pass &= check(abs(conf - 0.91) < 0.01, f"Confidence matches (got {conf:.4f})")

    name2, conf2 = store.get_name_for_speaker("SPEAKER_02")
    all_pass &= check(name2 is None, "None-mapped speaker returns None")
    all_pass &= check(conf2 == 0.0, "None-mapped speaker returns 0.0 confidence")

    # Reload from disk
    store2 = ParticipantStore(data_dir=TEST_DIR)
    loaded = store2.load_speaker_mapping()
    all_pass &= check("SPEAKER_00" in loaded, "SPEAKER_00 persisted across store instances")
    all_pass &= check(loaded["SPEAKER_00"]["name"] == "Alice Johnson", "Name persisted correctly")
    all_pass &= check(loaded["SPEAKER_01"]["name"] == "Bob Smith", "SPEAKER_01 persisted correctly")

    shutil.rmtree(TEST_DIR)
    return all_pass


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [
        ("order_heuristic_mapping",      test_order_heuristic_mapping()),
        ("transcript_name_mapping",       test_transcript_name_mapping()),
        ("fusion_prefers_higher_conf",    test_fusion_prefers_higher_confidence()),
        ("build_mapping_end_to_end",      test_build_mapping_end_to_end()),
        ("save_load_speaker_mapping",     test_save_load_speaker_mapping()),
    ]

    print("\n══ Summary ══")
    all_ok = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        all_ok &= passed

    sys.exit(0 if all_ok else 1)
