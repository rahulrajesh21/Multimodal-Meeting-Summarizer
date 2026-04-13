"""
RoME: Role-aware Multimodal Meeting Summarizer
Professional Interface
"""

import streamlit as st
import time
import threading
from typing import Optional, List, Dict
import os
import sys
import json
from datetime import datetime
import subprocess
import numpy as np
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_capture import AudioCapture
from src.live_transcription import LiveTranscriber
from src.text_analysis import TextAnalyzer, RoleBasedHighlightScorer
from src.visual_analysis import VisualAnalyzer
from src.audio_analysis import AudioTonalAnalyzer, load_audio_file, LIBROSA_AVAILABLE
from src.fusion_layer import FusionLayer, SegmentFeatures
from src.feedback_manager import FeedbackManager
from src.participant_store import ParticipantStore
from src.role_hierarchy import get_fallback_weights, get_role_description
from src.speaker_identifier import SpeakerIdentifier, LOW_CONFIDENCE_THRESHOLD

# Try to import Temporal Graph Memory
try:
    from src.temporal_graph_memory import TemporalGraphMemory
    TEMPORAL_MEMORY_AVAILABLE = True
except ImportError:
    TEMPORAL_MEMORY_AVAILABLE = False

from src.video_processing import VideoSummarizer

# Page config
st.set_page_config(
    page_title="RoME System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    defaults = {
        'transcript_text': '',
        'video_audio_path': None,
        'is_transcribing': False,
        'text_analyzer': None,
        'audio_tonal_analyzer': None,
        'visual_analyzer': None,
        'fusion_layer': None,
        'cached_audio_data': None,
        'role_mapping': {},
        'role_embeddings': {},
        'transcriber': None,
        'device': 'cpu',
        'scored_segments': None,
        'ai_summary': None,
        'temporal_memory': None,
        'current_meeting_id': None,
        'meeting_topics': [],
        'meeting_decisions': [],
        'meeting_action_items': [],
        'feedback_manager': None,
        'fusion_weights': None,
        'participant_store': None,
        'per_participant_highlights': {},
        # Speaker auto-mapping state
        'speaker_identifier': None,
        'raw_speaker_mapping': {},        # {SPEAKER_XX: (name, confidence)} from SpeakerIdentifier
        'confirmed_speaker_mapping': {},  # after user approves / overrides
        'raw_diarized_segments': [],      # raw segment list from diarization (with SPEAKER_XX labels)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_ffmpeg():
    st.error("FFmpeg is not installed or not found in system PATH.")
    st.stop()

# Helper functions
def get_text_analyzer():
    if st.session_state.text_analyzer is None:
        with st.spinner("Initializing Text Analyzer..."):
            st.session_state.text_analyzer = TextAnalyzer(device=st.session_state.device)
    return st.session_state.text_analyzer

def get_audio_analyzer():
    if st.session_state.audio_tonal_analyzer is None and LIBROSA_AVAILABLE:
        with st.spinner("Initializing Audio Analyzer..."):
            st.session_state.audio_tonal_analyzer = AudioTonalAnalyzer(sample_rate=16000)
    return st.session_state.audio_tonal_analyzer

def get_highlight_scorer():
    analyzer = get_text_analyzer()
    temporal_mem = get_temporal_memory()
    return RoleBasedHighlightScorer(text_analyzer=analyzer, temporal_memory=temporal_mem)

def get_temporal_memory():
    """Get or initialize the Temporal Graph Memory."""
    if not TEMPORAL_MEMORY_AVAILABLE:
        return None
    if st.session_state.temporal_memory is None:
        with st.spinner("Initializing Temporal Graph Memory..."):
            text_analyzer = get_text_analyzer()
            memory_dir = os.path.join(os.path.dirname(__file__), 'data', 'temporal_memory')
            st.session_state.temporal_memory = TemporalGraphMemory(
                text_analyzer=text_analyzer,
                storage_path=memory_dir
            )
    return st.session_state.temporal_memory

def parse_transcript_to_segments(transcript_text: str) -> List[Dict]:
    """Parse transcript text into segment dictionaries."""
    import re
    segments = []
    lines = transcript_text.strip().split('\n')
    pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\s*(?:\[(.*?)\])?\s*(.*)')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            timestamp_str = match.group(1)
            speaker = match.group(2)
            text = match.group(3).strip()
            if not text:
                continue
            parts = timestamp_str.split(':')
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            segments.append({
                'start': float(seconds),
                'end': float(seconds + 3),
                'text': text,
                'speaker': speaker
            })
    return segments

def get_feedback_manager():
    """Get or initialize the FeedbackManager."""
    if st.session_state.feedback_manager is None:
        st.session_state.feedback_manager = FeedbackManager()
        st.session_state.fusion_weights = st.session_state.feedback_manager.load_weights()
    return st.session_state.feedback_manager

def get_participant_store():
    """Get or initialize the ParticipantStore (no LLM pipeline at startup to keep it fast)."""
    if st.session_state.participant_store is None:
        st.session_state.participant_store = ParticipantStore(data_dir="data")
    return st.session_state.participant_store


def get_speaker_identifier() -> SpeakerIdentifier:
    """Get or initialize the SpeakerIdentifier (lazy — no heavy models loaded at startup)."""
    if st.session_state.speaker_identifier is None:
        hf_token = os.getenv("HF_TOKEN")
        st.session_state.speaker_identifier = SpeakerIdentifier(
            hf_token=hf_token,
            device=st.session_state.device,
        )
        # Load any previously saved voice prints
        vp_path = os.path.join(os.path.dirname(__file__), "data", "voice_prints.npz")
        st.session_state.speaker_identifier.load_voice_prints(vp_path)
    return st.session_state.speaker_identifier

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("RoME Configuration")
    
    st.subheader("Model Settings")
    # model_type is selectable now
    
    model_size = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1
    )
    
    device = st.selectbox(
        "Compute Device",
        ["cpu", "cuda", "mps"],
        index=0
    )
    st.session_state.device = device
    
    backend = st.selectbox(
        "Transcription Backend",
        ["faster_whisper", "huggingface"],
        index=0,
        help="faster-whisper (CTranslate2) is optimized for CPU/CUDA. Hugging Face supports MPS but may be slower."
    )
    st.session_state.backend = backend
    
    enable_diarization = st.checkbox(
        "Enable Speaker Diarization",
        value=True,
        help="Identify unique speakers in the audio."
    )
    
    st.divider()
    st.info("System Ready")
    
    # Temporal Memory Controls in Sidebar
    if TEMPORAL_MEMORY_AVAILABLE:
        st.divider()
        st.subheader("Temporal Memory")
        
        temporal_mem = get_temporal_memory()
        if temporal_mem:
            stats = temporal_mem.get_statistics()
            st.metric("Meetings", stats['meetings'])
            st.metric("Topics Tracked", stats['topics'])
            st.metric("Decisions", stats['decisions'])
            st.metric("Action Items", stats['action_items'])
            
            if st.button("Save Memory", use_container_width=True):
                temporal_mem.save()
                st.success("Memory saved!")
            
            if st.button("Clear Memory", use_container_width=True):
                st.session_state.temporal_memory = None
                st.session_state.current_meeting_id = None
                st.rerun()
    else:
        st.divider()
        st.warning("Temporal Memory unavailable")

# --- Main Interface ---
st.title("RoME: Role-aware Multimodal Meeting Summarizer")
st.markdown("### End-to-End Pipeline")

# Section 1: Input & Preprocessing
st.header("1. Input & Preprocessing")
st.markdown("Upload video, extract audio, transcribe, and map speakers.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Meeting Video", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_file is not None:
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_audio_path = file_path
        st.success(f"Video loaded: {uploaded_file.name}")

with col2:
    if st.button("Process Video (Transcribe & Diarize)", disabled=not st.session_state.video_audio_path):
        progress_bar = st.progress(0, text="Initializing...")

        try:
            # 1. Initialize Transcriber
            hf_token = os.getenv("HF_TOKEN")

            transcriber = LiveTranscriber(
                model_size=model_size,
                hf_token=hf_token,
                device=device,
                enable_diarization=enable_diarization,
                backend=st.session_state.backend
            )
            st.session_state.transcriber = transcriber

            # 2. Extract Audio
            progress_bar.progress(20, text="Extracting Audio Stream...")
            target_fps = 16000
            cmd = [
                'ffmpeg', '-i', st.session_state.video_audio_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(target_fps), '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # 3. Transcribe & Diarize — store raw segments (SPEAKER_XX labels intact)
            progress_bar.progress(40, text="Transcribing and Diarizing...")
            segments_generator = transcriber.transcribe_full_audio(result.stdout, target_fps)

            full_text = ""
            segments = list(segments_generator)

            # ── Store raw diarized segments for auto-mapping step ──
            st.session_state.raw_diarized_segments = segments
            # Reset any previous mapping so the new auto-mapping section is shown fresh
            st.session_state.raw_speaker_mapping = {}
            st.session_state.confirmed_speaker_mapping = {}

            # Build initial transcript with raw SPEAKER_XX labels
            for i, segment in enumerate(segments):
                start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                text = segment['text']
                speaker = segment.get('speaker')

                if speaker:
                    line = f"[{start_time}] [{speaker}] {text}\n\n"
                else:
                    line = f"[{start_time}] {text}\n\n"
                full_text += line

                prog = 40 + int(50 * (i / len(segments)))
                progress_bar.progress(min(prog, 90), text=f"Processing segment {i+1}...")

            st.session_state.transcript_text = full_text

            # 4. Cache Audio for Fusion
            if LIBROSA_AVAILABLE:
                st.session_state.cached_audio_data = load_audio_file(
                    st.session_state.video_audio_path, sample_rate=16000
                )

            progress_bar.progress(100, text="Complete")
            st.success("✅ Transcription complete — see Auto Speaker Mapping below.")

        except Exception as e:
            st.error(f"Processing Error: {e}")


# ── Auto Speaker Mapping UI ──────────────────────────────────────────────────
if st.session_state.transcript_text:

    with st.expander("🔎 Auto Speaker Mapping", expanded=True):
        pstore_map = get_participant_store()
        registered = pstore_map.list_participants()  # [{display_name, role, ...}]
        registered_names = [p["display_name"] for p in registered]

        # Detect unique SPEAKER_XX labels in the raw segments
        raw_segs = st.session_state.raw_diarized_segments
        unique_speakers = []
        for seg in raw_segs:
            spk = seg.get("speaker", "")
            if spk and spk.startswith("SPEAKER_") and spk not in unique_speakers:
                unique_speakers.append(spk)

        has_speakers = bool(unique_speakers)

        if not has_speakers:
            st.info("No diarization labels found in this transcript (diarization may have been disabled).")

        else:
            col_auto1, col_auto2 = st.columns([2, 1])

            with col_auto1:
                # ── Run auto-detection ────────────────────────────────────────
                if st.button("🤖 Auto-Detect Speaker Names", type="primary"):
                    if not registered_names:
                        st.warning("No participants registered yet — register them in the Participant Management section below first.")
                    else:
                        with st.spinner("Running speaker identification..."):
                            si = get_speaker_identifier()
                            audio_np = st.session_state.cached_audio_data
                            if audio_np is None and st.session_state.video_audio_path:
                                # Load audio if not cached
                                import subprocess as _sp
                                _cmd = ['ffmpeg', '-i', st.session_state.video_audio_path,
                                        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000',
                                        '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-']
                                _res = _sp.run(_cmd, stdout=_sp.PIPE, stderr=_sp.PIPE)
                                audio_np = np.frombuffer(_res.stdout, dtype=np.int16).astype(np.float32) / 32768.0

                            mapping = si.build_mapping(
                                diarization_segments=raw_segs,
                                audio_array=audio_np if audio_np is not None else np.zeros(16000, dtype=np.float32),
                                sample_rate=16000,
                                registered_participants=registered_names,
                                transcript_text=st.session_state.transcript_text,
                            )
                            st.session_state.raw_speaker_mapping = mapping
                            st.success(f"Auto-mapping complete for {len(mapping)} speakers.")
                            st.rerun()

            with col_auto2:
                st.caption(f"**{len(unique_speakers)}** speaker(s) detected · **{len(registered_names)}** participant(s) registered")

            # ── Display mapping table ─────────────────────────────────────────
            raw_map = st.session_state.raw_speaker_mapping

            st.markdown("---")
            st.markdown("**Review & Confirm Speaker Assignments**")
            st.caption("✅ High confidence · ⚠️ Low confidence (check) · ❓ Unmatched — use the dropdown to assign")

            # Build dropdowns: ['— Unassigned —'] + registered names
            dropdown_options = ["— Unassigned —"] + registered_names

            override_map: Dict[str, str] = {}   # will be populated by selectboxes

            for spk in unique_speakers:
                name, conf = raw_map.get(spk, (None, 0.0))
                if name and conf >= LOW_CONFIDENCE_THRESHOLD:
                    badge = "✅"
                    conf_str = f"{conf*100:.0f}%"
                elif name:
                    badge = "⚠️"
                    conf_str = f"{conf*100:.0f}%"
                else:
                    badge = "❓"
                    conf_str = "N/A"

                row_cols = st.columns([0.15, 0.25, 0.15, 0.45])
                with row_cols[0]:
                    st.markdown(f"**{spk}**")
                with row_cols[1]:
                    st.markdown(f"{badge} `{name or 'Unknown'}`")
                with row_cols[2]:
                    st.caption(f"Conf: {conf_str}")
                with row_cols[3]:
                    # Pre-select the auto-detected name when confidence is reasonable
                    if name and name in dropdown_options:
                        default_idx = dropdown_options.index(name)
                    else:
                        default_idx = 0
                    chosen = st.selectbox(
                        f"Assign {spk}",
                        options=dropdown_options,
                        index=default_idx,
                        key=f"spk_override_{spk}",
                        label_visibility="collapsed",
                    )
                    override_map[spk] = chosen if chosen != "— Unassigned —" else None

            st.markdown("---")

            if st.button("✅ Apply Speaker Mapping & Update Transcript", type="primary"):
                # Build confirmed mapping from overrides
                confirmed: Dict[str, tuple] = {}
                for spk in unique_speakers:
                    resolved = override_map.get(spk)
                    orig_conf = raw_map.get(spk, (None, 0.0))[1] if raw_map.get(spk) else 0.0
                    confirmed[spk] = (resolved, orig_conf if resolved else 0.0)

                st.session_state.confirmed_speaker_mapping = confirmed

                # Persist to ParticipantStore
                pstore_map.save_speaker_mapping(confirmed)

                # Rebuild transcript with resolved names
                updated_transcript = ""
                for seg in raw_segs:
                    start_time = time.strftime('%H:%M:%S', time.gmtime(seg['start']))
                    text = seg.get('text', '')
                    raw_spk = seg.get('speaker', '')
                    resolved_name, _c = confirmed.get(raw_spk, (None, 0.0))
                    display_name = resolved_name or raw_spk or "Unknown"
                    if raw_spk:
                        updated_transcript += f"[{start_time}] [{display_name}] {text}\n\n"
                    else:
                        updated_transcript += f"[{start_time}] {text}\n\n"

                st.session_state.transcript_text = updated_transcript

                # Update role_mapping for backward compat + generate embeddings
                new_role_mapping = {}
                analyzer = get_text_analyzer()
                for spk, (name, _) in confirmed.items():
                    if name:
                        # Look up the person's role from ParticipantStore
                        profile = pstore_map.get_participant(name)
                        role_label = f"{name} ({profile['role']})" if profile else name
                        new_role_mapping[spk] = role_label
                        emb = analyzer.get_embedding(role_label)
                        if emb is not None:
                            st.session_state.role_embeddings[role_label] = emb

                st.session_state.role_mapping = new_role_mapping
                st.success(f"✅ Speaker mapping applied! Transcript updated with {len([v for v in override_map.values() if v])} resolved names.")
                st.rerun()

        # ── Preview current transcript ────────────────────────────────────────
        with st.expander("📄 Current Transcript Preview", expanded=False):
            st.text_area(
                "Transcript",
                value=st.session_state.transcript_text,
                height=200,
                disabled=True,
                key="transcript_preview_area",
            )


    # --- Participant Management ---
    with st.expander("👥 Participant Management", expanded=False):
        pstore = get_participant_store()

        st.markdown("Register participants so the system uses their real role and weights during analysis.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Add Participant**")
            p_name  = st.text_input("Name", placeholder="Alice Johnson", key="pm_name")
            p_role  = st.text_input("Role", placeholder="CTO / Client - ACME / Freelance Advisor", key="pm_role")
            p_dept  = st.text_input("Department (optional)", key="pm_dept")
            p_ext   = st.checkbox("External participant", key="pm_ext")

            if st.button("Register Participant", disabled=not (p_name and p_role)):
                # Try to attach LLM pipeline if already loaded
                llm_pipeline = None
                if 'llm_summarizer' in st.session_state and st.session_state.llm_summarizer.is_ready:
                    llm_pipeline = st.session_state.llm_summarizer.pipeline
                    pstore.llm_pipeline = llm_pipeline

                profile = pstore.add_participant(
                    name=p_name, role=p_role,
                    is_external=p_ext, department=p_dept
                )
                source = profile.get("weight_source", "formula")
                st.success(
                    f"✅ Registered **{p_name}** as *{p_role}* "
                    f"(weights via {'🤖 LLM' if source == 'llm' else '📐 formula'})"
                )
                st.json(profile["weights"])

            # Unknown role expander
            with st.expander("🔧 Role not in system? Register custom authority"):
                cr_role = st.text_input("Role title", key="cr_role")
                cr_auth = st.slider("Authority score", 0.0, 1.0, 0.5, key="cr_auth")
                if st.button("Save custom role", key="cr_save"):
                    pstore.register_custom_role(cr_role, cr_auth)
                    st.success(f"Saved '{cr_role}' with authority {cr_auth:.2f}")

        with col_p2:
            st.markdown("**Registered Participants**")
            participants = pstore.list_participants()
            if participants:
                for p in participants:
                    w = p["weights"]
                    desc = get_role_description(p["role"], p.get("is_external", False))
                    st.markdown(
                        f"**{p['display_name']}** — {p['role']}  \n"
                        f"_{desc}_  \n"
                        f"`S:{w['semantic']:.2f}  T:{w['tonal']:.2f}  R:{w['role']:.2f}  H:{w['temporal']:.2f}`  \n"
                        f"Source: *{p.get('weight_source','formula')}*"
                    )
                    st.divider()
            else:
                st.caption("No participants registered yet.")

st.divider()

# Section 2: Core Processing
st.header("2. Core Processing (Fusion)")
st.markdown("Configure multimodal fusion to identify role-relevant highlights.")

col_core1, col_core2 = st.columns(2)

with col_core1:
    target_role = st.selectbox(
        "Target Audience Role",
        ["Product Manager", "Developer", "Designer", "QA Engineer", "Executive", "CEO", "CTO"],
        index=0
    )
    
    fusion_mode = st.radio(
        "Fusion Mode",
        ["Heuristic (Auto Weights)", "Neural (Transformer Model)"],
        horizontal=True
    )
    
    focus_query = st.text_input("Custom Focus Area (Optional)", placeholder="e.g., budget, deadlines, technical debt")

with col_core2:
    # Auto-weight badge — resolve from participant store or fallback
    pstore = get_participant_store()
    profile = pstore.get_participant(target_role)
    if profile:
        auto_weights = profile["weights"]
        weight_source_label = f"🟢 Profile: {profile['display_name']}"
        role_desc = get_role_description(profile["role"], profile.get("is_external", False))
    else:
        auto_weights = get_fallback_weights(target_role, is_external=False)
        weight_source_label = "🟡 Heuristic (no profile found)"
        role_desc = get_role_description(target_role)

    st.markdown(f"**Auto Weights** — {weight_source_label}")
    st.caption(role_desc)
    st.info(
        f"Semantic: **{auto_weights['semantic']:.2f}** · "
        f"Tonal: **{auto_weights['tonal']:.2f}** · "
        f"Role gate: **{auto_weights['role']:.2f}** · "
        f"Temporal: **{auto_weights['temporal']:.2f}**"
    )

if st.button("Run Multimodal Analysis", type="primary"):
    if not st.session_state.transcript_text:
        st.warning("Please complete Step 1 first.")
    else:
        with st.spinner("Running Fusion Analysis..."):
            try:
                text_analyzer = get_text_analyzer()
                audio_analyzer = get_audio_analyzer()
                temporal_memory = get_temporal_memory()
                pstore = get_participant_store()
                
                fusion_layer = FusionLayer(
                    text_analyzer=text_analyzer,
                    audio_analyzer=audio_analyzer,
                    weights=auto_weights,
                    temporal_memory=temporal_memory,
                    participant_store=pstore,
                )
                if st.session_state.role_embeddings:
                    fusion_layer.set_role_embeddings(st.session_state.role_embeddings)
                
                segments = parse_transcript_to_segments(st.session_state.transcript_text)
                
                if fusion_mode.startswith("Neural"):
                     scored_segments = fusion_layer.score_segments_contextual(
                        segments, target_role, st.session_state.cached_audio_data,
                        sample_rate=16000, focus_query=focus_query, use_ml=True
                    )
                else:
                    scored_segments = fusion_layer.score_segments(
                        segments, target_role, st.session_state.cached_audio_data,
                        sample_rate=16000, focus_query=focus_query
                    )
                
                st.session_state.scored_segments = scored_segments
                
                # Auto-ingest into Temporal Memory (if a meeting is active)
                if temporal_memory and st.session_state.current_meeting_id:
                    with st.spinner("⏳ Saving to Temporal Memory..."):
                        temporal_memory.ingest_meeting_results(
                            meeting_id=st.session_state.current_meeting_id,
                            scored_segments=scored_segments,
                            importance_threshold=0.4
                        )
                        temporal_memory.save()
                        # Invalidate thread cache so next render recomputes
                        try:
                            from src.thread_detector import ThreadDetector
                            if hasattr(temporal_memory, '_thread_detector'):
                                temporal_memory._thread_detector.invalidate_cache()
                        except Exception:
                            pass
                    st.success(
                        f"✅ Meeting saved to Temporal Memory — "
                        f"{len(st.session_state.scored_segments)} segments ingested"
                    )
                speakers_seen = {s.speaker for s in scored_segments if s.speaker}
                for spk in speakers_seen:
                    st.caption(pstore.get_ui_badge(spk) + f" — {spk}")
                
                st.success(f"Analysis Complete. Processed {len(segments)} segments.")
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

        # --- Feedback Section ---
        if st.session_state.scored_segments:
            st.divider()
            st.subheader("Review & Refine (Online Learning)")
            st.markdown("Provide feedback on top segments to adapt the model to your preferences.")
            
            # Show top 5 segments
            sorted_segs = sorted(st.session_state.scored_segments, key=lambda x: x.fused_score, reverse=True)[:5]
            
            for i, seg in enumerate(sorted_segs):
                with st.container():
                    c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
                    with c1:
                        st.markdown(f"**#{i+1}**")
                    with c2:
                        st.markdown(f"_{seg.text}_")
                        score_line = (
                            f"Score: {seg.fused_score:.2f} "
                            f"(S:{seg.semantic_score:.2f}, T:{seg.tonal_score:.2f}, "
                            f"R:{seg.role_relevance:.2f}, H:{seg.temporal_context_score:.2f})"
                        )
                        st.caption(score_line)
                        # Show thread badge if this segment is part of a recurring thread
                        if getattr(seg, 'thread_info', None):
                            ann = seg.thread_info.get('annotation', '')
                            if ann:
                                st.caption(ann)
                    with c3:
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button("Like", key=f"like_{i}"):
                                fm = get_feedback_manager()
                                scores = {
                                    'semantic': seg.semantic_score,
                                    'tonal': seg.tonal_score,
                                    'role': seg.role_relevance,
                                    'temporal': seg.temporal_context_score
                                }
                                fm.log_feedback(seg.text, scores, 'like')
                                new_weights = fm.update_weights(
                                    st.session_state.fusion_weights, scores, 1.0,
                                    speaker_name=seg.speaker,
                                    participant_store=get_participant_store(),
                                )
                                st.session_state.fusion_weights = new_weights
                                st.toast("Feedback recorded. Model updated.")
                                time.sleep(1)
                                st.rerun()
                        with cc2:
                            if st.button("Dislike", key=f"dislike_{i}"):
                                fm = get_feedback_manager()
                                scores = {
                                    'semantic': seg.semantic_score,
                                    'tonal': seg.tonal_score,
                                    'role': seg.role_relevance,
                                    'temporal': seg.temporal_context_score
                                }
                                fm.log_feedback(seg.text, scores, 'dislike')
                                new_weights = fm.update_weights(
                                    st.session_state.fusion_weights, scores, -1.0,
                                    speaker_name=seg.speaker,
                                    participant_store=get_participant_store(),
                                )
                                st.session_state.fusion_weights = new_weights
                                st.toast("Feedback recorded. Model updated.")
                                time.sleep(1)
                                st.rerun()
                    st.divider()

# Advanced: Training
with st.expander("Advanced: Model Training"):
    st.markdown("Train the Neural Fusion model on the current transcript.")
    if st.button("Train Model"):
        if st.session_state.scored_segments:
            with st.spinner("Training..."):
                try:
                    from src.train_fusion import FusionTrainer
                    from src.llm_summarizer import LLMSummarizer
                    
                    if 'llm_summarizer' not in st.session_state:
                        st.session_state.llm_summarizer = LLMSummarizer()
                    
                    trainer = FusionTrainer(st.session_state.llm_summarizer)
                    loss = trainer.train_step(st.session_state.scored_segments, epochs=5)
                    st.success(f"Training Complete. Loss: {loss:.4f}")
                except Exception as e:
                    st.error(f"Training Error: {e}")
        else:
            st.warning("Run Analysis first.")

st.divider()

# Section 3: Generation
st.header("3. Generation & Results")
st.markdown("Generate role-specific video highlights and text summaries.")

col_gen1, col_gen2 = st.columns(2)

with col_gen1:
    st.subheader("Video Highlight")
    if st.button("Generate Video Digest"):
        if not st.session_state.scored_segments:
            st.warning("Run Analysis first.")
        elif not st.session_state.video_audio_path:
            st.warning("No video loaded.")
        else:
            with st.spinner("Generating Video..."):
                try:
                    scorer = get_highlight_scorer()
                    summarizer = VideoSummarizer(scorer)
                    
                    # Prepare segments
                    seg_data = [{
                        'start': s.start_time, 'end': s.end_time,
                        'score': s.fused_score, 'text': s.text
                    } for s in st.session_state.scored_segments]
                    
                    # Smooth
                    ranges = summarizer.filter_and_smooth(seg_data, threshold=0.4, min_gap=2.0, padding=1.0)
                    
                    if ranges:
                        output_filename = f"summary_{target_role.replace(' ', '_')}_{int(time.time())}.mp4"
                        output_path = os.path.join("exports", output_filename)
                        os.makedirs("exports", exist_ok=True)
                        
                        result_path = summarizer.create_summary_video(
                            st.session_state.video_audio_path, ranges, output_path, crossfade_duration=0.5
                        )
                        st.video(result_path)
                        st.success(f"Video Generated: {output_filename}")
                    else:
                        st.warning("No significant highlights found.")
                except Exception as e:
                    st.error(f"Generation Error: {e}")

with col_gen2:
    st.subheader("Text Summary (Per Participant)")
    top_n = st.slider("Top segments per person", 3, 10, 5, key="top_n_summary")

    # LLM status badge + reload
    from src.llm_summarizer import LLMSummarizer as _LLMSummarizer
    _DEFAULT_MODEL = "facebook/bart-large-cnn"

    # Auto-reinitialize if wrong model is cached
    if 'llm_summarizer' in st.session_state:
        cached = st.session_state.llm_summarizer
        if getattr(cached, 'model_name', None) != _DEFAULT_MODEL:
            del st.session_state['llm_summarizer']   # force reload

    col_llm1, col_llm2 = st.columns([3, 1])
    with col_llm1:
        if 'llm_summarizer' in st.session_state:
            st.caption(f"LLM: {st.session_state.llm_summarizer.status()}")
        else:
            st.caption(f"LLM: ⏳ `{_DEFAULT_MODEL}` — will load on first click")
    with col_llm2:
        if st.button("🔄 Reload LLM", key="reload_llm"):
            if 'llm_summarizer' in st.session_state:
                del st.session_state['llm_summarizer']
            st.rerun()

    if st.button("Generate Text Summary"):
        if not st.session_state.scored_segments:
            st.warning("Run Analysis first.")
        else:
            with st.spinner("Generating per-participant summaries..."):
                try:
                    from src.llm_summarizer import LLMSummarizer
                    if 'llm_summarizer' not in st.session_state:
                        st.session_state.llm_summarizer = LLMSummarizer()
                    llm = st.session_state.llm_summarizer

                    pstore = get_participant_store()
                    all_segs = st.session_state.scored_segments

                    # Group segments by speaker
                    from collections import defaultdict
                    speaker_segs = defaultdict(list)
                    for seg in all_segs:
                        spk = seg.speaker or "Unknown"
                        speaker_segs[spk].append(seg)

                    per_participant_summaries = {}
                    per_participant_highlights = {}

                    # Role-specific semantic queries for better re-ranking
                    ROLE_QUERIES = {
                        "ceo":              "strategic direction, company priorities, executive decisions, vision",
                        "cto":              "technical strategy, engineering decisions, architecture, R&D priorities",
                        "cfo":              "budget, financial decisions, cost, revenue, fiscal direction",
                        "product manager":  "product roadmap, feature priorities, user needs, sprint planning",
                        "developer":        "technical implementation, code, bugs, technical debt, estimates",
                        "designer":         "UX, user experience, design decisions, interface, visual feedback",
                        "analyst":          "data, metrics, KPIs, performance, reporting, analysis",
                        "infrastructure":   "infrastructure, deployment, reliability, DevOps, system stability",
                        "quality":          "quality, testing, QA, defects, release readiness",
                    }

                    def _role_query(role_str: str) -> str:
                        low = role_str.lower()
                        for key, query in ROLE_QUERIES.items():
                            if key in low:
                                return query
                        return "important decisions, key contributions, significant remarks"

                    text_analyzer = get_text_analyzer()

                    for spk, segs in speaker_segs.items():
                        profile = pstore.get_participant(spk)
                        role = profile["role"] if profile else spk

                        # Role-specific semantic re-ranking
                        role_query = focus_query.strip() if focus_query and focus_query.strip() else _role_query(role)
                        
                        def _score_seg(seg, query=role_query):
                            """Compute role-specific semantic similarity for re-ranking."""
                            try:
                                if text_analyzer:
                                    q_emb = text_analyzer.get_embedding(query)
                                    s_emb = seg.text_embedding if seg.text_embedding is not None \
                                        else text_analyzer.get_embedding(seg.text)
                                    if q_emb is not None and s_emb is not None:
                                        import numpy as _np
                                        cos = float(_np.dot(q_emb, s_emb) / (
                                            _np.linalg.norm(q_emb) * _np.linalg.norm(s_emb) + 1e-8
                                        ))
                                        return (cos + 1) / 2  # 0-1
                            except Exception:
                                pass
                            return seg.fused_score  # fallback

                        scored = sorted(segs, key=_score_seg, reverse=True)
                        top_segs = scored[:top_n]
                        top_segs_sorted = sorted(top_segs, key=lambda s: s.start_time)

                        per_participant_highlights[spk] = top_segs_sorted

                        highlight_text = "\n".join(
                            f"[{s.start_time:.1f}s] {s.text}" for s in top_segs_sorted
                        )

                        if llm.is_ready and highlight_text.strip():
                            summary = llm.summarize(
                                highlight_text,
                                role=role,
                                focus=role_query,
                            )
                        else:
                            summary = "\n".join(
                                f"• [{s.start_time:.1f}s] {s.text[:120]}{'…' if len(s.text) > 120 else ''}"
                                for s in top_segs_sorted
                            )

                        per_participant_summaries[spk] = summary


                    st.session_state.ai_summary = per_participant_summaries
                    st.session_state.per_participant_highlights = per_participant_highlights

                except Exception as e:
                    st.error(f"Summary Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display per-participant summaries
    if st.session_state.ai_summary and isinstance(st.session_state.ai_summary, dict):
        pstore = get_participant_store()
        for spk, summary in st.session_state.ai_summary.items():
            profile = pstore.get_participant(spk)
            role    = profile["role"] if profile else "Unknown Role"
            w       = profile["weights"] if profile else {}

            with st.expander(f"{'🟢' if profile else '🟡'}  {spk}  —  {role}", expanded=True):
                if w:
                    st.caption(
                        f"S:{w.get('semantic',0):.2f}  "
                        f"T:{w.get('tonal',0):.2f}  "
                        f"R:{w.get('role',0):.2f}  "
                        f"H:{w.get('temporal',0):.2f}  "
                        f"| source: {profile.get('weight_source','formula')}"
                    )
                st.info(summary)

                # Show which segments contributed
                highlights = st.session_state.get("per_participant_highlights", {}).get(spk, [])
                if highlights:
                    with st.expander("📌 Top segments used", expanded=False):
                        for seg in highlights:
                            score_bar = "█" * int(seg.fused_score * 10) + "░" * (10 - int(seg.fused_score * 10))
                            st.markdown(
                                f"**[{seg.start_time:.1f}s – {seg.end_time:.1f}s]** "
                                f"`{score_bar}` `{seg.fused_score:.2f}`  \n{seg.text}"
                            )

    # Legacy: plain string summary from older sessions
    elif st.session_state.ai_summary and isinstance(st.session_state.ai_summary, str):
        st.info(st.session_state.ai_summary)


# Section 4: Temporal Graph Memory (Cross-Meeting Context)
if TEMPORAL_MEMORY_AVAILABLE:
    st.divider()
    st.header("🧠 4. Temporal Graph Memory")
    st.markdown(
        "Track how decisions, issues, and topics **evolve across multiple meetings**. "
        "The system automatically detects recurring threads and highlights them in summaries."
    )

    temporal_mem = get_temporal_memory()

    if temporal_mem:
        # --- Tab layout ---
        tab_hist, tab_threads, tab_search, tab_actions = st.tabs([
            "📊 Meeting History",
            "🧵 Cross-Meeting Threads",
            "🔍 Semantic Search",
            "✅ Open Action Items",
        ])

        # ================================================================
        # TAB 1: Meeting History
        # ================================================================
        with tab_hist:
            st.subheader("Meeting History & Management")

            col_new, col_list = st.columns([1, 1])

            with col_new:
                with st.expander("➕ Register Current Meeting",
                                 expanded=not st.session_state.current_meeting_id):
                    meeting_title = st.text_input(
                        "Meeting Title", placeholder="Weekly Standup — Feb 20",
                        key="tm_title"
                    )
                    meeting_participants = st.text_input(
                        "Participants (comma-separated)", placeholder="Alice, Bob, Charlie",
                        key="tm_participants"
                    )
                    meeting_tags = st.text_input(
                        "Tags (comma-separated)", placeholder="standup, planning, q1",
                        key="tm_tags"
                    )
                    meeting_date_input = st.date_input(
                        "Meeting Date", key="tm_date"
                    )

                    if st.button("Create & Set as Current", type="primary", key="tm_create"):
                        if meeting_title:
                            from datetime import datetime as _dt
                            parts = [p.strip() for p in meeting_participants.split(',') if p.strip()]
                            tags  = [t.strip() for t in meeting_tags.split(',') if t.strip()]
                            m_date = _dt.combine(meeting_date_input, _dt.min.time()) if meeting_date_input else None

                            mid = temporal_mem.create_meeting(
                                title=meeting_title,
                                date=m_date,
                                participants=parts,
                                tags=tags,
                            )
                            st.session_state.current_meeting_id = mid
                            temporal_mem.save()
                            st.success(f"✅ Meeting created (ID: {mid}). It is now the active meeting.")
                            st.rerun()
                        else:
                            st.warning("Please enter a meeting title.")

                # Manual ingest if auto-ingest was skipped (e.g., no meeting was active during analysis)
                if st.session_state.scored_segments and st.session_state.current_meeting_id:
                    if st.button("📥 Re-import segments to active meeting", key="tm_reimport"):
                        with st.spinner("Ingesting (clearing stale data first)..."):
                            result = temporal_mem.ingest_meeting_results(
                                meeting_id=st.session_state.current_meeting_id,
                                scored_segments=st.session_state.scored_segments,
                                importance_threshold=0.4,
                                clear=True,
                            )
                            temporal_mem.save()
                        st.success(
                            f"Imported: {len(result['segments'])} segments, "
                            f"{len(result['topics'])} topics, "
                            f"{len(result['decisions'])} decisions, "
                            f"{len(result['action_items'])} action items."
                        )
                        st.rerun()

            with col_list:
                st.markdown("**All Recorded Meetings**")
                meetings = temporal_mem.get_all_meetings()
                if meetings:
                    for m in meetings:
                        title = m.content  # content is plain string
                        date_str = m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else "Unknown date"
                        is_active = (m.id == st.session_state.current_meeting_id)
                        badge = " 🟢 **[Active]**" if is_active else ""
                        tags_str = ", ".join(m.metadata.get('tags', []))
                        tag_line = f"  `{tags_str}`" if tags_str else ""
                        # Count topics (entity type='topic' with events in this meeting)
                        n_topics = sum(
                            1 for ent_id in temporal_mem.entities_by_type.get('topic', [])
                            if any(
                                temporal_mem.events[eid].meeting_id == m.id
                                for eid in temporal_mem.events_by_entity.get(ent_id, [])
                                if eid in temporal_mem.events
                            )
                        )
                        # Count events (segments) in this meeting
                        n_segs = len(temporal_mem.events_by_meeting.get(m.id, []))
                        status_icon = "📊" if n_segs > 0 else "⚪"
                        st.markdown(
                            f"{status_icon} **{title}**{badge}  \n"
                            f"_{date_str}_{tag_line}  \n"
                            f"&nbsp;&nbsp;&nbsp;{n_segs} events · {n_topics} topics"
                        )
                        if not is_active:
                            if st.button(f"Set as active", key=f"tm_activate_{m.id}"):
                                st.session_state.current_meeting_id = m.id
                                st.rerun()
                        st.divider()
                else:
                    st.caption("No meetings recorded yet. Create one above or run analysis with an active meeting.")

            # Active meeting summary
            if st.session_state.current_meeting_id:
                active_mid = st.session_state.current_meeting_id
                active_mtg_data = temporal_mem.meetings.get(active_mid)
                if active_mtg_data:
                    st.markdown("---")
                    st.subheader(f"📄 Active: {active_mtg_data['title']}")
                    summary = temporal_mem.get_meeting_summary(active_mid)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Events", len(summary.get("segments", [])))
                    c2.metric("Topics", len(summary.get("topics", [])))
                    c3.metric("Decisions", len(summary.get("decisions", [])))
                    c4.metric("Action Items", len(summary.get("action_items", [])))

        # ================================================================
        # TAB 2: Cross-Meeting Threads
        # ================================================================
        with tab_threads:
            st.subheader("🧵 Recurring Topic Threads Across Meetings")
            st.markdown(
                "Topics that appear in **two or more meetings** are automatically grouped into threads. "
                "This is how the system detects when something discussed on Wednesday resurfaces on Sunday."
            )

            # --- Diagnostic: show topic count per meeting so user knows if ready ---
            meetings_with_topics = []
            for m in temporal_mem.get_all_meetings():
                n_topics = sum(
                    1 for ent_id in temporal_mem.entities_by_type.get('topic', [])
                    if any(
                        temporal_mem.events[eid].meeting_id == m.id
                        for eid in temporal_mem.events_by_entity.get(ent_id, [])
                        if eid in temporal_mem.events
                    )
                )
                meetings_with_topics.append((m.content, n_topics))

            if meetings_with_topics:
                total_meetings_with_data = sum(1 for _, t in meetings_with_topics if t > 0)

                if total_meetings_with_data < 2:
                    st.warning(
                        f"💡 **{total_meetings_with_data} of {len(meetings_with_topics)} meetings have been processed** "
                        f"(have transcribed+analysed video with an active meeting set). "
                        f"Thread detection needs ≥ 2 processed meetings."
                    )
                    st.markdown("**Topic count per meeting:**")
                    for m_title, n_t in meetings_with_topics:
                        icon = "✅" if n_t > 0 else "❌"
                        st.markdown(f"- {icon} **{m_title}**: {n_t} topics")
                    st.markdown(
                        "**How to fix:** Set the other meeting as active, then run ‘Run Multimodal Analysis’ "
                        "on a video — segments will auto-save. Or use ‘Re-import segments’ in Meeting History."
                    )

            col_thr_btn, col_thr_thresh = st.columns([1, 2])
            with col_thr_btn:
                detect_btn = st.button("🔍 Detect Threads", type="primary", key="detect_threads_btn")
            with col_thr_thresh:
                similarity_threshold = st.slider(
                    "Similarity threshold", min_value=0.30, max_value=0.90,
                    value=0.50, step=0.05,
                    help="Lower = more threads found (keyword overlap); Higher = only near-identical topics linked",
                    key="thread_sim_threshold"
                )

            if detect_btn or st.session_state.get("_threads_detected"):
                with st.spinner("Clustering topics across meetings..."):
                    threads = temporal_mem.find_cross_meeting_threads(
                        min_meetings=2,
                        similarity_threshold=similarity_threshold
                    )
                st.session_state["_threads_detected"] = True

                if not threads:
                    st.info(
                        f"📅 No recurring threads detected at threshold {similarity_threshold:.2f}. "
                        f"Try lowering the similarity threshold slider."
                    )
                else:
                    st.success(f"🎓 Found **{len(threads)}** recurring thread(s) across meetings.")
                    for thread in threads:
                        first_date = thread['first_seen'][:10]
                        last_date  = thread['last_seen'][:10]
                        n_mtgs     = thread['meeting_count']
                        kws        = ", ".join(thread.get('keywords', [])[:5])
                        label      = thread['label']

                        with st.expander(
                            f"🧵 {label}   —   {n_mtgs} meetings   —   {first_date} → {last_date}",
                        ):
                            if kws:
                                st.caption(f"Keywords: `{kws}`")

                            # Timeline
                            st.markdown("**Timeline:**")
                            for i, appearance in enumerate(thread['appearances']):
                                icon = "🟡" if i == 0 else ("🔴" if i == len(thread['appearances']) - 1 else "🔵")
                                appr_date = appearance['date'][:16].replace('T', ' ')
                                st.markdown(
                                    f"{icon} **{appr_date}** — *{appearance['meeting_title']}*  \n"
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;{appearance['topic']}"
                                )
                                if appearance.get('keywords'):
                                    st.caption("&nbsp;&nbsp;&nbsp;&nbsp;`" + ", ".join(appearance['keywords'][:4]) + "`")

        # ================================================================
        # TAB 3: Semantic Search
        # ================================================================
        with tab_search:
            st.subheader("🔍 Search Across All Meetings")
            st.markdown("Semantic search: results are ranked by **meaning similarity**, not just keyword match.")

            search_query = st.text_input(
                "Search query",
                placeholder="security issue in backend, budget approval, launch date...",
                key="tm_search_query"
            )

            if search_query and st.button("Search", key="tm_search_btn"):
                with st.spinner("Searching memory..."):
                    results = temporal_mem.get_context_for_text(search_query, top_k=12)

                if not results:
                    st.info("No matching items found. Try different keywords, or process more meetings first.")
                else:
                    st.markdown(f"**{len(results)} results** for `{search_query}`:")
                    for item in results:
                        node_type = item.get('type', 'unknown')
                        score     = item.get('similarity', 0)
                        date_str  = (item.get('timestamp') or '')[:10]

                        type_icons = {
                            'decision': '🟢',
                            'action_item': '✅',
                            'topic': '📌',
                            'segment': '💬',
                        }
                        icon = type_icons.get(node_type, '•')

                        col_r1, col_r2 = st.columns([0.1, 0.9])
                        with col_r1:
                            st.markdown(icon)
                        with col_r2:
                            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                            st.markdown(f"`{bar}` **{score:.0%}** match  —  `{node_type}` — {date_str}")
                            content = (
                                item.get('text') or item.get('name') or
                                item.get('topic') or 'N/A'
                            )
                            st.caption(content[:200])
                            if item.get('speaker'):
                                st.caption(f"Speaker: {item['speaker']}")
                        st.divider()

        # ================================================================
        # TAB 4: Open Action Items
        # ================================================================
        with tab_actions:
            st.subheader("✅ Open Action Items (All Meetings)")

            open_items = temporal_mem.get_open_action_items()

            if not open_items:
                st.info("No open action items tracked yet. Run analysis with an active meeting to auto-extract them.")
            else:
                priority_order = {'high': 0, 'medium': 1, 'low': 2}
                open_items.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))

                # Filter controls
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    filter_priority = st.selectbox(
                        "Filter by priority", ["all", "high", "medium", "low"],
                        key="tm_filter_priority"
                    )
                with filter_col2:
                    filter_assignee = st.text_input("Filter by assignee", key="tm_filter_assignee")

                displayed = [
                    item for item in open_items
                    if (filter_priority == "all" or item.get('priority') == filter_priority)
                    and (not filter_assignee or filter_assignee.lower() in (item.get('assignee') or '').lower())
                ]

                st.caption(f"Showing {len(displayed)} of {len(open_items)} open items")

                priority_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}

                for item in displayed:
                    icon = priority_icons.get(item.get('priority', 'medium'), '•')
                    assignee_str = item.get('assignee') or 'Unassigned'
                    meeting_str  = item.get('meeting_title', 'Unknown')
                    date_str     = (item.get('date') or '')[:10]

                    col_a, col_b = st.columns([0.05, 0.95])
                    with col_a:
                        st.markdown(icon)
                    with col_b:
                        st.markdown(f"**{item['action'][:140]}**")
                        st.caption(f"👤 {assignee_str}  —  🗓 {date_str}  —  📍 {meeting_str}")

                        if st.button("Mark done", key=f"tm_done_{item['action_id']}"):
                            temporal_mem.update_action_status(item['action_id'], 'done')
                            temporal_mem.save()
                            st.rerun()
                    st.divider()

