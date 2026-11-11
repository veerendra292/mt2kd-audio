import os, sys, time
from pathlib import Path
import streamlit as st
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="MT2KD Audio Studio",
    page_icon="🎙️",
    layout="wide"
)

# Load MT2KD Engine
@st.cache_resource
def load_engine():
    from model_engine import MT2KDEngine
    model_dir = Path(__file__).resolve().parent / "model"
    return MT2KDEngine(model_dir=str(model_dir))

with st.spinner("Loading MT2KD Multi-Task Conformer (19M parameters)..."):
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Header
st.title("🎙️ MT2KD Multimodal Audio Studio")
st.markdown("Unified Multi-Task Conformer: Automatic Speech Recognition (ASR), Audio Tagging (AT), and Speaker Verification (SV).")

# 3 Dedicated Task Tabs
tab1, tab2, tab3 = st.tabs([
    "🗣️ Task 1: Speech Recognition (ASR)",
    "🏷️ Task 2: Audio Tagging (AT)",
    "👥 Task 3: Speaker Verification (SV)"
])

# --- Tab 1: ASR ---
with tab1:
    st.subheader("Task 1: Automatic Speech Recognition")
    st.caption("Upload spoken audio or speech to transcribe.")
    asr_file = st.file_uploader("Choose a speech audio file (WAV)", type=["wav", "flac"], key="asr_upload")
    if asr_file is not None:
        st.audio(asr_file)
        if st.button("Transcribe Speech", type="primary", key="asr_btn"):
            with st.spinner("Transcribing..."):
                t0 = time.perf_counter()
                audio_bytes = asr_file.read()
                transcript = engine.transcribe(audio_bytes)
                latency = round((time.perf_counter() - t0) * 1000, 1)
            
            st.success("Transcription Complete!")
            st.text_area("Transcript:", value=transcript, height=120)
            st.metric("Inference Latency", f"{latency} ms")

# --- Tab 2: Audio Tagging ---
with tab2:
    st.subheader("Task 2: Sound Event Classification (Audio Tagging)")
    st.caption("Upload any acoustic sound clip (instruments, animals, ambient sounds).")
    at_file = st.file_uploader("Choose a sound audio file (WAV)", type=["wav", "flac"], key="at_upload")
    if at_file is not None:
        st.audio(at_file)
        if st.button("Classify Audio Event", type="primary", key="at_btn"):
            with st.spinner("Classifying audio events..."):
                audio_bytes = at_file.read()
                tags = engine.audio_tag(audio_bytes, top_k=5)
            
            st.success(f"Top Sound Event: **{tags[0]['label']}** ({tags[0]['score']*100:.1f}% confidence)")
            
            # Display horizontal bars
            for t in tags:
                st.write(f"**{t['label']}** ({t['score']*100:.1f}%)")
                st.progress(float(t['score']))

# --- Tab 3: Speaker Verification ---
with tab3:
    st.subheader("Task 3: Speaker Verification (Voice Comparison)")
    st.caption("Upload two voice recordings to verify whether they belong to the same person.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Voice Recording 1 (Reference)")
        sv_file1 = st.file_uploader("Upload Voice 1", type=["wav", "flac"], key="sv1")
        if sv_file1:
            st.audio(sv_file1)
            
    with col2:
        st.write("##### Voice Recording 2 (Comparison)")
        sv_file2 = st.file_uploader("Upload Voice 2", type=["wav", "flac"], key="sv2")
        if sv_file2:
            st.audio(sv_file2)
            
    if sv_file1 and sv_file2:
        if st.button("Compare Speakers", type="primary", key="sv_btn"):
            with st.spinner("Extracting 192-D speaker embeddings and comparing..."):
                b1 = sv_file1.read()
                b2 = sv_file2.read()
                ver = engine.verify(b1, b2, threshold=0.50)
                sim = ver["score"]
                is_match = ver["same_speaker"]
                
            if is_match:
                st.success(f"✅ **SAME SPEAKER (MATCH)** — Cosine Similarity: **{sim * 100:.1f}%**")
            else:
                st.error(f"❌ **DIFFERENT SPEAKER (NON-MATCH)** — Cosine Similarity: **{sim * 100:.1f}%**")
                
            st.progress(max(0.0, min(1.0, float(sim))))
            st.caption(f"Decision Threshold: 50.0% · Embedding Dimension: 192-D")
