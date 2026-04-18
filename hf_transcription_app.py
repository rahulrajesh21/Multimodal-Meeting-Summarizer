import streamlit as st
import os
import subprocess
import time
from typing import Dict, List
import sys
import requests
import io

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.live_transcription import LiveTranscriber

st.set_page_config(page_title="HF Transcription & Diarization", layout="wide")

# Initialize session state variables
if "video_audio_path" not in st.session_state:
    st.session_state.video_audio_path = None
if "raw_segments" not in st.session_state:
    st.session_state.raw_segments = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "speaker_mapping" not in st.session_state:
    st.session_state.speaker_mapping = {}

st.title("Hugging Face Transcription & Diarization")
st.markdown("Upload a media file, transcribe it using a Hugging Face Whisper model, perform speaker diarization, and manually map the speakers.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input("Hugging Face Token (HF_TOKEN)", type="password", value=os.environ.get("HF_TOKEN", ""))
    model_size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    device = st.selectbox("Compute Device", ["cpu", "cuda", "mps"], index=0)
    backend = st.selectbox("Backend", ["huggingface", "faster_whisper", "whisply"], index=2)
    st.divider()
    st.markdown("### Step 1: Upload Media")
    uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp4", "mp3", "wav", "m4a", "mov"])

if uploaded_file is not None:
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_audio_path = file_path
    st.success(f"Loaded: {uploaded_file.name}")

if st.button("Transcribe & Diarize", type="primary"):
    if not st.session_state.video_audio_path:
        st.error("Please upload a file first!")
    elif not hf_token:
        st.error("Please provide a Hugging Face Token for diarization!")
    else:
        with st.spinner("Extracting audio and initializing transcriber..."):
            os.environ["HF_TOKEN"] = hf_token # Set it for pyannote

            # 1. Extract Audio
            target_fps = 16000
            cmd = [
                'ffmpeg', '-y', '-i', st.session_state.video_audio_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(target_fps), '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio_bytes = result.stdout

            # 2. Setup Transcriber
            transcriber = LiveTranscriber(
                model_size=model_size,
                hf_token=hf_token,
                device=device,
                enable_diarization=True,
                backend=backend
            )

        with st.spinner("Transcribing and Diarizing... this might take a while."):
            # 3. Process
            segments_generator = transcriber.transcribe_full_audio(audio_bytes, target_fps)
            segments = list(segments_generator)
            st.session_state.raw_segments = segments

            # Extract unique speakers
            for seg in segments:
                spk = seg.get("speaker")
                if spk and spk not in st.session_state.speaker_mapping:
                    st.session_state.speaker_mapping[spk] = ""
            
            st.success("Transcription complete!")

# --- Speaker Mapping UI ---
if st.session_state.raw_segments:
    st.divider()
    st.header("Step 2: Speaker Mapping")
    
    unique_speakers = sorted([s for s in list(st.session_state.speaker_mapping.keys()) if s])
    
    if not unique_speakers:
        st.warning("No speakers detected by diarization.")
    else:
        st.markdown("Assign real names to the detected system identifiers (e.g. `SPEAKER_00`).")
        cols = st.columns(min(len(unique_speakers), 4))
        
        updated_mapping = {}
        for i, spk in enumerate(unique_speakers):
            with cols[i % len(cols)]:
                current_val = st.session_state.speaker_mapping.get(spk, "")
                updated_mapping[spk] = st.text_input(f"Name for **{spk}**", value=current_val, key=f"map_{spk}")
        
        if st.button("Apply Mapping & Update Transcript"):
            st.session_state.speaker_mapping = updated_mapping
            st.success("Mapping applied!")

    # Generate the finalized transcript based on the mapping
    st.header("Step 3: Final Transcript")
    full_text = ""
    for seg in st.session_state.raw_segments:
        start_time = time.strftime('%H:%M:%S', time.gmtime(seg.get('start', 0)))
        raw_spk = seg.get('speaker', '')
        text = seg.get('text', '')
        
        # Resolve speaker name
        display_name = st.session_state.speaker_mapping.get(raw_spk, raw_spk)
        if not display_name:
            display_name = raw_spk
            
        if display_name:
            full_text += f"[{start_time}] [{display_name}] {text}\n\n"
        else:
            full_text += f"[{start_time}] {text}\n\n"

    st.text_area("Transcript Output", value=full_text, height=400, disabled=True)

    st.header("Step 4: Save to Teams Media Server")
    meeting_name = st.text_input("Meeting Name", value="Imported Transcript from App")
    
    if st.button("Save Transcript to Teams Server", type="secondary"):
        with st.spinner("Saving to Teams Media Server (http://localhost:8001)..."):
            try:
                # 1. Create a meeting
                meeting_resp = requests.post(
                    "http://localhost:8001/v1.0/me/onlineMeetings",
                    data={"subject": meeting_name}
                )
                if not meeting_resp.ok:
                    st.error(f"Failed to create meeting: {meeting_resp.text}")
                else:
                    mid = meeting_resp.json().get("id")
                    
                    # 2. Format as WebVTT
                    vtt_lines = ["WEBVTT", ""]
                    for seg in st.session_state.raw_segments:
                        start_time = time.strftime('%H:%M:%S', time.gmtime(seg.get('start', 0))) + ".000"
                        end_time = time.strftime('%H:%M:%S', time.gmtime(seg.get('end', 0))) + ".000"
                        
                        raw_spk = seg.get('speaker', '')
                        display_name = st.session_state.speaker_mapping.get(raw_spk, raw_spk) or "Unknown"
                        text = seg.get('text', '')
                        
                        vtt_lines.append(f"{start_time} --> {end_time}")
                        vtt_lines.append(f"<v {display_name}>{text}</v>")
                        vtt_lines.append("")
                        
                    vtt_content = "\n".join(vtt_lines)
                    
                    # 3. Upload transcript
                    files = {"file": ("transcript.vtt", io.BytesIO(vtt_content.encode("utf-8")), "text/vtt")}
                    tx_resp = requests.post(
                        f"http://localhost:8001/v1.0/me/onlineMeetings/{mid}/transcripts",
                        files=files
                    )
                    
                    if tx_resp.ok:
                        st.success(f"Successfully saved transcript to Teams Media Server! Meeting ID: `{mid}`")
                    else:
                        st.error(f"Failed to upload transcript: {tx_resp.text}")
            except requests.ConnectionError:
                st.error("Could not connect to the Teams Media Server. Is it running on http://localhost:8001?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
