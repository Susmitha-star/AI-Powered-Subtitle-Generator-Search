import streamlit as st
import whisper
import numpy as np
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
from pydub import AudioSegment

# Load the subtitle embeddings (Already Generated)
with open("subtitle_embeddings.pkl", "rb") as f:
    subtitle_data = pickle.load(f)

# Initialize ChromaDB for fast search
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="subtitles")

# Load Whisper Model
whisper_model = whisper.load_model("base")

# Load Sentence Transformer Model (For Semantic Search)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to Transcribe Audio (Using Whisper AI)
def transcribe_audio(audio_file):
    # Save uploaded file to a temp location
    temp_input_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_input_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Convert to WAV (Whisper prefers WAV)
    sound = AudioSegment.from_file(temp_input_path)  # Automatically detects format
    temp_wav_path = temp_input_path + ".wav"
    sound.export(temp_wav_path, format="wav")

    # Transcribe using Whisper
    result = whisper_model.transcribe(temp_wav_path, word_timestamps=True)

    # Remove temp files
    os.remove(temp_input_path)
    os.remove(temp_wav_path)

    # Convert output to subtitle format (SRT)
    subtitles = []
    for i, segment in enumerate(result["segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        start_time_str = format_time(start_time)
        end_time_str = format_time(end_time)

        subtitles.append(f"{i+1}\n{start_time_str} --> {end_time_str}\n{text}\n")

    return "\n".join(subtitles)  # Return as an SRT string

# Function to format time for SRT (HH:MM:SS,MS)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

st.set_page_config(page_title="Subtitle Generator & Search", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🎬 AI-Powered Subtitle Generator & Search</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an audio file to generate subtitles and search within existing subtitles.</p>
    """,
    unsafe_allow_html=True
)

# Upload Section
st.markdown("### 🔊 Upload an Audio Clip")
uploaded_audio = st.file_uploader("Supported formats: MP3, WAV, FLAC, AAC, OGG", type=["mp3", "wav", "flac", "aac", "ogg", "m4a"])

if uploaded_audio:
    st.audio(uploaded_audio, format="audio/mp3")
    
    with st.spinner("📝 Generating subtitles..."):
        subtitle_srt = transcribe_audio(uploaded_audio)
    
    # Display subtitles
    st.markdown("### 📜 Generated Subtitles")
    st.text_area("", subtitle_srt, height=250)
    
    # Save and Provide Download
    subtitle_filename = "generated_subtitles.srt"
    with open(subtitle_filename, "w", encoding="utf-8") as f:
        f.write(subtitle_srt)
    
    with open(subtitle_filename, "rb") as f:
        st.download_button("📥 Download Subtitles", data=f, file_name="subtitles.srt", mime="text/plain")
    
    # Cleanup
    os.remove(subtitle_filename)
