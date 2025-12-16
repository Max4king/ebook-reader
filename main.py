import os
import subprocess

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

VOICE_MAP = {
    "Puck": "Male - Upbeat & Energetic",
    "Charon": "Male - Deep & Authoritative",
    "Fenrir": "Male - Friendly & Excitable",
    "Aoede": "Female - Sophisticated & Articulate",
    "Kore": "Female - Calm & Professional",
    "Zephyr": "Female - Bright & Youthful",
    "Orus": "Male - Mature & Resonant",
    "Leda": "Female - Composed & Authoritative",
    "Enceladus": "Male - Soft & Breathy",
}


def extract_text_from_pdf_bytes(file_bytes):
    """Extract raw text from PDF using pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-", "-"],
            input=file_bytes,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return None


def clean_text_with_gemini(raw_text, model_name="gemini-2.5-flash"):
    """
    Uses Gemini to remove headers/footers.
    We generally default to Flash for this because it's cheap and text-only.
    """
    prompt = """
    You are an expert ebook formatter. 
    1. Remove page numbers, headers, footers, and copyright notices.
    2. Join hyphenated words split across lines.
    3. Return ONLY the clean text. Do not summarize.
    
    Text to clean:
    """
    try:
        response = client.models.generate_content(
            model=model_name, contents=prompt + raw_text[:30000]
        )
        return response.text
    except Exception as e:
        st.error(f"Cleaning Error: {e}")
        return raw_text


def generate_gemini_tts(text, model_name, voice_name, voice_direction=None):
    """
    Generates audio using the selected Gemini 2.5 model.

    Args:
        text: The text to convert to speech
        model_name: The Gemini model to use
        voice_name: The voice persona to use
        voice_direction: Optional instruction for how the voice should sound
    """
    # Default prompt if none provided
    default_prompt = "Read this text in a natural, engaging voice suitable for an audiobook. Speak clearly and at a comfortable pace."

    # Combine text with voice direction
    if voice_direction and voice_direction.strip():
        full_content = f"{voice_direction.strip()}\n\n{text}"
    else:
        full_content = f"{default_prompt}\n\n{text}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_content,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )

        # Concatenate audio parts
        audio_data = b""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_data += part.inline_data.data
        return audio_data

    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


# --- UI LAYOUT ---
st.set_page_config(page_title="Gemini 2.5 Ebook Narrator", layout="wide")

st.sidebar.title("⚙️ Settings")

# 1. Model Selector
model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    ("models/gemini-2.5-flash-preview-tts", "models/gemini-2.5-pro-preview-tts"),
    index=0,
    help="Flash is faster & cheaper. Pro has better emotional range.",
)

# 2. Voice Selector
voice_choice = st.sidebar.selectbox(
    "Choose Voice Persona",
    options=list(VOICE_MAP.keys()),
    format_func=lambda x: f"{x} ({VOICE_MAP[x]})",
    index=0,
)

# 3. Voice Direction Input
voice_direction = st.sidebar.text_area(
    "Voice Direction (Optional)",
    placeholder="Describe how you want the voice to sound. Example: 'Speak in a calm, soothing tone perfect for bedtime stories' or 'Read with excitement and enthusiasm for this adventure story.'",
    height=100,
    help="Provide specific instructions for the voice style. If left empty, a default audiobook-appropriate prompt will be used."
)
st.title(f"📖 Ebook Narrator ({model_choice})")

if not api_key:
    st.warning("⚠️ GEMINI_API_KEY not found. Please set it in your environment.")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file and api_key:
    # --- STEP 1: EXTRACT ---
    with st.spinner("Extracting text..."):
        if uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
        else:
            raw_text = uploaded_file.getvalue().decode("utf-8")

    if raw_text:
        # Layout: Left for Text, Right for Audio
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Text Processing")
            with st.expander("Show Raw Text", expanded=False):
                st.text_area("Raw", raw_text[:2000], height=150)

            # Clean Text Button
            if st.button("Clean Text (Remove Headers/Footers)"):
                with st.spinner("Cleaning text with Gemini Flash..."):
                    # We always use Flash for cleaning to save money,
                    # unless you want Pro to do the cleaning too.
                    clean_text = clean_text_with_gemini(raw_text[:10000])
                    st.session_state["clean_text"] = clean_text
                    st.rerun()

        # --- STEP 2: GENERATE ---
        with col2:
            st.subheader("2. Audio Generation")

            if "clean_text" in st.session_state:
                st.text_area("Cleaned Text", st.session_state["clean_text"], height=300)

                st.write(f"**Selected Model:** `{model_choice}`")

                if st.button("Generate Audio"):
                    with st.spinner(f"Synthesizing audio using {model_choice}..."):
                        audio_bytes = generate_gemini_tts(
                            st.session_state["clean_text"], model_choice, voice_choice, voice_direction
                        )

                        if audio_bytes:
                            st.success("Audio Generated!")
                            st.audio(audio_bytes, format="audio/wav")
                            st.download_button(
                                "Download WAV",
                                audio_bytes,
                                "audiobook.wav",
                                "audio/wav",
                            )
            else:
                st.info("👈 Please clean the text first.")
