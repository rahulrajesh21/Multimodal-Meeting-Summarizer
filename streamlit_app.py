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
        'meeting_topics': [],
        'meeting_decisions': [],
        'meeting_action_items': [],
        'feedback_manager': None,
        'fusion_weights': None
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
            memory_path = os.path.join(os.path.dirname(__file__), 'data', 'temporal_memory.json')
            st.session_state.temporal_memory = TemporalGraphMemory(
                text_analyzer=text_analyzer,
                storage_path=memory_path
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
            
            # 3. Transcribe & Diarize
            progress_bar.progress(40, text="Transcribing and Diarizing...")
            segments_generator = transcriber.transcribe_full_audio(result.stdout, target_fps)
            
            full_text = ""
            segments = list(segments_generator)
            
            for i, segment in enumerate(segments):
                start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                text = segment['text']
                speaker = segment.get('speaker')
                
                if speaker:
                    mapped_name = st.session_state.role_mapping.get(speaker, speaker)
                    line = f"[{start_time}] [{mapped_name}] {text}\n\n"
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
            st.success("Preprocessing Complete")
            
        except Exception as e:
            st.error(f"Processing Error: {e}")

# Role Mapping
if st.session_state.transcript_text:
    with st.expander("Speaker Role Mapping", expanded=True):
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            default_mapping = '{\n  "SPEAKER_00": "Product Manager",\n  "SPEAKER_01": "Developer",\n  "SPEAKER_02": "Designer"\n}'
            mapping_json = st.text_area("Role Map (JSON)", value=default_mapping, height=150)
            if st.button("Apply Role Mapping"):
                try:
                    mapping = json.loads(mapping_json)
                    st.session_state.role_mapping = mapping
                    
                    # Generate Embeddings
                    analyzer = get_text_analyzer()
                    for speaker_id, role in mapping.items():
                        emb = analyzer.get_embedding(role)
                        if emb is not None:
                            st.session_state.role_embeddings[role] = emb
                    
                    # Update Transcript
                    updated = st.session_state.transcript_text
                    for speaker_id, role in mapping.items():
                        updated = updated.replace(f"[{speaker_id}]", f"[{role}]")
                    st.session_state.transcript_text = updated
                    st.success("Roles Mapped and Embeddings Generated")
                except Exception as e:
                    st.error(f"Mapping Error: {e}")
        
        with col_map2:
            st.text_area("Current Transcript", value=st.session_state.transcript_text, height=200, disabled=True)

st.divider()

# Section 2: Core Processing
st.header("2. Core Processing (Fusion)")
st.markdown("Configure multimodal fusion to identify role-relevant highlights.")

col_core1, col_core2 = st.columns(2)

with col_core1:
    target_role = st.selectbox(
        "Target Audience Role",
        ["Product Manager", "Developer", "Designer", "QA Engineer", "Executive"],
        index=0
    )
    
    fusion_mode = st.radio(
        "Fusion Mode",
        ["Heuristic (Manual Weights)", "Neural (Transformer Model)"],
        horizontal=True
    )
    
    focus_query = st.text_input("Custom Focus Area (Optional)", placeholder="e.g., budget, deadlines, technical debt")

with col_core2:
    st.markdown("**Fusion Weights (Heuristic Mode)**")
    
    # Load default weights from FeedbackManager if available
    fm = get_feedback_manager()
    current_weights = st.session_state.fusion_weights or fm.default_weights
    
    w_semantic = st.slider("Semantic (Text)", 0.0, 1.0, current_weights.get('semantic', 0.4))
    w_tonal = st.slider("Tonal (Audio)", 0.0, 1.0, current_weights.get('tonal', 0.15))
    w_role = st.slider("Role (Context)", 0.0, 1.0, current_weights.get('role', 0.25))
    w_temporal = st.slider("Temporal (History)", 0.0, 1.0, current_weights.get('temporal', 0.2))

if st.button("Run Multimodal Analysis", type="primary"):
    if not st.session_state.transcript_text:
        st.warning("Please complete Step 1 first.")
    else:
        with st.spinner("Running Fusion Analysis..."):
            try:
                text_analyzer = get_text_analyzer()
                audio_analyzer = get_audio_analyzer()
                temporal_memory = get_temporal_memory()
                
                # Weights
                total = w_semantic + w_tonal + w_role + w_temporal
                weights = {
                    'semantic': w_semantic/total if total > 0 else 0.4,
                    'tonal': w_tonal/total if total > 0 else 0.15,
                    'role': w_role/total if total > 0 else 0.25,
                    'temporal': w_temporal/total if total > 0 else 0.2
                }
                
                fusion_layer = FusionLayer(
                    text_analyzer=text_analyzer,
                    audio_analyzer=audio_analyzer,
                    weights=weights,
                    temporal_memory=temporal_memory
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
                        st.caption(f"Score: {seg.fused_score:.2f} (S:{seg.semantic_score:.2f}, T:{seg.tonal_score:.2f}, R:{seg.role_relevance:.2f}, H:{seg.temporal_context_score:.2f})")
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
                                new_weights = fm.update_weights(st.session_state.fusion_weights, scores, 1.0)
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
                                new_weights = fm.update_weights(st.session_state.fusion_weights, scores, -1.0)
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
                        st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                    
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
    st.subheader("Text Summary")
    if st.button("Generate Text Summary"):
        if not st.session_state.transcript_text:
            st.warning("No transcript.")
        else:
            with st.spinner("Generating Summary..."):
                try:
                    from src.llm_summarizer import LLMSummarizer
                    if 'llm_summarizer' not in st.session_state:
                        st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                    
                    summary = st.session_state.llm_summarizer.summarize(
                        st.session_state.transcript_text, target_role, focus_query or "key points"
                    )
                    st.session_state.ai_summary = summary
                except Exception as e:
                    st.error(f"Summary Error: {e}")
    
    if st.session_state.ai_summary:
        st.info(st.session_state.ai_summary)

# Section 4: Temporal Graph Memory (Cross-Meeting Context)
if TEMPORAL_MEMORY_AVAILABLE:
    st.divider()
    st.header("4. Temporal Graph Memory")
    st.markdown("Track decisions, action items, and topics across meetings for cross-meeting continuity.")
    
    temporal_mem = get_temporal_memory()
    
    if temporal_mem:
        col_tm1, col_tm2 = st.columns(2)
        
        with col_tm1:
            st.subheader("Meeting Management")
            
            # Create new meeting
            with st.expander("Create New Meeting", expanded=not st.session_state.current_meeting_id):
                meeting_title = st.text_input("Meeting Title", placeholder="Weekly Standup - Q1 Review")
                meeting_participants = st.text_input("Participants (comma-separated)", placeholder="Alice, Bob, Charlie")
                meeting_tags = st.text_input("Tags (comma-separated)", placeholder="standup, planning, q1")
                
                if st.button("Create Meeting", type="primary"):
                    if meeting_title:
                        participants = [p.strip() for p in meeting_participants.split(',') if p.strip()]
                        tags = [t.strip() for t in meeting_tags.split(',') if t.strip()]
                        
                        meeting_id = temporal_mem.create_meeting(
                            title=meeting_title,
                            participants=participants,
                            tags=tags
                        )
                        st.session_state.current_meeting_id = meeting_id
                        st.success(f"Meeting created: {meeting_id[:8]}...")
                        st.rerun()
                    else:
                        st.warning("Please enter a meeting title.")
            
            # Current meeting info
            if st.session_state.current_meeting_id:
                meeting_node = temporal_mem.nodes.get(st.session_state.current_meeting_id)
                if meeting_node:
                    st.info(f"**Current Meeting:** {meeting_node.content.get('title', 'Untitled')}")
                    
                    # Add segments from current transcript
                    if st.session_state.scored_segments and st.button("Import Segments to Memory"):
                        with st.spinner("Importing segments..."):
                            for seg in st.session_state.scored_segments:
                                temporal_mem.add_segment(
                                    meeting_id=st.session_state.current_meeting_id,
                                    text=seg.text,
                                    start_time=seg.start_time,
                                    end_time=seg.end_time,
                                    speaker=seg.speaker,
                                    importance_score=seg.fused_score
                                )
                            temporal_mem.save()
                            st.success(f"Imported {len(st.session_state.scored_segments)} segments!")
            
            # List past meetings
            st.markdown("---")
            st.markdown("**Past Meetings:**")
            meetings = temporal_mem.get_all_meetings()
            if meetings:
                for m in meetings[-5:]:  # Show last 5
                    title = m.content.get('title', 'Untitled')
                    date = m.timestamp[:10] if m.timestamp else 'Unknown'
                    title = m.content.get('title', 'Untitled')
                    date = m.timestamp[:10] if m.timestamp else 'Unknown'
                    if st.button(f"{title} ({date})", key=f"load_{m.node_id}"):
                        st.session_state.current_meeting_id = m.node_id
                        st.rerun()
            else:
                st.caption("No meetings yet.")
        
        with col_tm2:
            st.subheader("Track Items")
            
            if st.session_state.current_meeting_id:
                # Add Decision
                with st.expander("Add Decision"):
                    decision_text = st.text_area("Decision", placeholder="We decided to postpone the launch to Q2", key="decision_text")
                    decision_owner = st.text_input("Owner", placeholder="Product Manager", key="decision_owner")
                    
                    if st.button("Add Decision"):
                        if decision_text:
                            temporal_mem.add_decision(
                                meeting_id=st.session_state.current_meeting_id,
                                decision_text=decision_text,
                                owner=decision_owner or None
                            )
                            temporal_mem.save()
                            st.success("Decision added!")
                        else:
                            st.warning("Please enter decision text.")
                
                # Add Action Item
                with st.expander("Add Action Item"):
                    action_text = st.text_area("Action Item", placeholder="Update the roadmap document", key="action_text")
                    action_assignee = st.text_input("Assignee", placeholder="Alice", key="action_assignee")
                    action_due = st.date_input("Due Date", key="action_due")
                    action_priority = st.selectbox("Priority", ["low", "medium", "high"], index=1, key="action_priority")
                    
                    if st.button("Add Action Item"):
                        if action_text:
                            temporal_mem.add_action_item(
                                meeting_id=st.session_state.current_meeting_id,
                                action_text=action_text,
                                assignee=action_assignee or None,
                                due_date=action_due.isoformat() if action_due else None,
                                priority=action_priority
                            )
                            temporal_mem.save()
                            st.success("Action item added!")
                        else:
                            st.warning("Please enter action item text.")
                
                # Add Topic
                with st.expander("Add Topic"):
                    topic_name = st.text_input("Topic Name", placeholder="Q2 Launch Planning", key="topic_name")
                    topic_desc = st.text_area("Description", placeholder="Discussion about launch timeline", key="topic_desc")
                    
                    if st.button("Add Topic"):
                        if topic_name:
                            temporal_mem.add_topic(
                                meeting_id=st.session_state.current_meeting_id,
                                topic_name=topic_name,
                                description=topic_desc or None
                            )
                            temporal_mem.save()
                            st.success("Topic added!")
                        else:
                            st.warning("Please enter topic name.")
            else:
                st.info("Create or select a meeting to track items.")
        
        # Cross-Meeting Context View
        st.markdown("---")
        st.subheader("Cross-Meeting Context")
        
        col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
        
        with col_ctx1:
            st.markdown("**Recent Decisions**")
            all_decisions = [n for n in temporal_mem.nodes.values() 
                          if hasattr(n, 'type') and n.type.value == 'decision']
            if all_decisions:
                for d in sorted(all_decisions, key=lambda x: x.timestamp, reverse=True)[:5]:
                    st.markdown(f"• {d.content.get('text', 'No text')[:80]}...")
            else:
                st.caption("No decisions tracked yet.")
        
        with col_ctx2:
            st.markdown("**Pending Action Items**")
            all_actions = [n for n in temporal_mem.nodes.values() 
                         if hasattr(n, 'type') and n.type.value == 'action_item' 
                         and n.content.get('status') != 'completed']
            if all_actions:
                for a in sorted(all_actions, key=lambda x: x.content.get('priority', 'medium') == 'high', reverse=True)[:5]:
                    assignee = a.content.get('assignee', 'Unassigned')
                    priority = a.content.get('priority', 'medium')
                    marker = "[HIGH]" if priority == "high" else "[MED]" if priority == "medium" else "[LOW]"
                    st.markdown(f"{marker} [{assignee}] {a.content.get('text', 'No text')[:60]}...")
            else:
                st.caption("No pending action items.")
        
        with col_ctx3:
            st.markdown("**Active Topics**")
            all_topics = [n for n in temporal_mem.nodes.values() 
                        if hasattr(n, 'type') and n.type.value == 'topic']
            if all_topics:
                for t in sorted(all_topics, key=lambda x: len(x.edges), reverse=True)[:5]:
                    mentions = len([e for e in t.edges if e.edge_type.value == 'mentions'])
                    st.markdown(f"• **{t.content.get('name', 'Unknown')}** ({mentions} mentions)")
            else:
                st.caption("No topics tracked yet.")
        
        # Semantic Search in Memory
        st.markdown("---")
        st.subheader("Search Memory")
        search_query = st.text_input("Search across all meetings", placeholder="budget decisions, launch timeline...")
        
        if search_query and st.button("Search"):
            with st.spinner("Searching..."):
                results = temporal_mem.get_context_for_text(search_query, top_k=10)
                
                if results:
                    st.markdown(f"**Found {len(results)} relevant items:**")
                    for item in results:
                        node_type = item.get('type', 'unknown')
                        score = item.get('similarity', 0)
                        
                        if node_type == 'decision':
                            st.markdown(f"**Decision** (relevance: {score:.2f})")
                            st.caption(item.get('text', 'No text'))
                        elif node_type == 'action_item':
                            st.markdown(f"**Action Item** (relevance: {score:.2f})")
                            st.caption(f"{item.get('text', 'No text')} - {item.get('assignee', 'Unassigned')}")
                        elif node_type == 'topic':
                            st.markdown(f"**Topic** (relevance: {score:.2f})")
                            st.caption(item.get('name', 'Unknown'))
                        elif node_type == 'segment':
                            st.markdown(f"**Segment** (relevance: {score:.2f})")
                            st.caption(item.get('text', 'No text')[:150] + "...")
                else:
                    st.info("No matching items found.")
