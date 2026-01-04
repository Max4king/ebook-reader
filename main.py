import os
import subprocess
import logging
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ebook_reader.log")],
)
logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    logger.info("API key loaded successfully from environment")
    client: genai.Client = genai.Client(api_key=api_key)
else:
    logger.error("Failed to load GEMINI_API_KEY from environment")
    # exit
    sys.exit(1)
    # client = None

LANGUAGE_MAP = {
    "Thai": "ภาษาไทย",
    "English": "English",
    "Japanese": "日本語",
}

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


def chunk_text_aware(text, max_chunk_size=5000):
    """
    Split text into chunks using sentence boundaries and paragraph breaks.

    This function is text-aware and respects:
    1. Paragraph breaks (newlines)
    2. Sentence boundaries (., !, ?)
    3. Reasonable chunk sizes while avoiding mid-sentence splits

    Args:
        text: The text to chunk
        max_chunk_size: Maximum characters per chunk (soft limit)

    Returns:
        List of text chunks
    """
    import re

    logger.info(f"Starting text-aware chunking for {len(text)} characters")

    # First split by paragraphs (double newlines or single newlines)
    paragraphs = re.split(r"\n+", text.strip())

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph itself is too large, split by sentences
        if len(para) > max_chunk_size:
            # Split paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding this sentence would exceed limit, start new chunk
                if (
                    len(current_chunk) + len(sentence) + 1 > max_chunk_size
                    and current_chunk
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        else:
            # Check if adding this paragraph would exceed limit
            if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Text chunked into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i + 1}: {len(chunk)} characters")
    return chunks


def extract_text_from_pdf_bytes(file_bytes):
    """Extract raw text from PDF using pdftotext."""
    logger.info("Starting PDF text extraction using pdftotext")
    try:
        logger.info(f"Running pdftotext command with {len(file_bytes)} bytes of input")
        result = subprocess.run(
            ["pdftotext", "-layout", "-", "-"],
            input=file_bytes,
            capture_output=True,
            check=True,
        )
        extracted_length = len(result.stdout)
        logger.info(
            f"PDF text extraction completed successfully. Extracted {extracted_length} characters"
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        st.error(f"Error extracting PDF: {e}")
        return None


def clean_text_with_gemini(raw_text, model_name="gemini-3-flash-preview"):
    """
    Uses Gemini to remove headers/footers.

    Handles large texts by chunking them into smaller pieces using
    text-aware chunking that respects sentence boundaries and paragraphs.

    We generally default to Flash for this because it's cheap and text-only.
    """
    logger.info(f"Starting text cleaning with Gemini model: {model_name}")
    prompt = """
    You are an expert ebook formatter.
    1. Remove headers, footers, and copyright notices.
    2. IMPORTANT: Keep page numbers in the format [Page X] or (Page X) to maintain structure.
    3. Join hyphenated words split across lines.
    4. Return ONLY the clean text. Do not summarize.

    Text to clean:
    """
    try:
        # Check if text needs chunking
        if len(raw_text) > 15000:
            logger.info(
                f"Text length ({len(raw_text)} chars) exceeds 15k, using text-aware chunking"
            )
            chunks = chunk_text_aware(raw_text, max_chunk_size=10000)
            cleaned_chunks = []

            for i, chunk in enumerate(chunks):
                logger.info(
                    f"Cleaning chunk {i + 1}/{len(chunks)} ({len(chunk)} characters)"
                )
                response = client.models.generate_content(
                    model=model_name, contents=prompt + chunk
                )
                cleaned_chunk = response.text or ""
                cleaned_chunks.append(cleaned_chunk)
                logger.info(
                    f"Chunk {i + 1} cleaning completed. Output length: {len(cleaned_chunk)} characters"
                )

            # Combine all cleaned chunks
            cleaned_text = "\n\n".join(cleaned_chunks)
            logger.info(
                f"All chunks cleaned successfully. Total output length: {len(cleaned_text)} characters"
            )
            return cleaned_text
        else:
            # Text is small enough to process in one go
            text_to_process = raw_text
            input_length = len(text_to_process)
            logger.info(
                f"Calling Gemini API for text cleaning. Input length: {input_length} characters"
            )
            response = client.models.generate_content(
                model=model_name, contents=prompt + text_to_process
            )
            cleaned_text = response.text or ""
            cleaned_length = len(cleaned_text)
            logger.info(
                f"Text cleaning completed successfully. Output length: {cleaned_length} characters"
            )
            return cleaned_text
    except Exception as e:
        logger.error(f"Text cleaning failed with error: {e}")
        st.error(f"Cleaning Error: {e}")
        return raw_text


def translate_text_with_gemini(
    text, target_language, model_name="gemini-3-flash-preview"
):
    """
    Translates text to the target language using Gemini.

    Handles large texts by chunking them into smaller pieces using
    text-aware chunking that respects sentence boundaries and paragraphs.
    Uses parallel processing for faster translation of multiple chunks.

    Args:
        text: The text to translate
        target_language: The language to translate to (e.g., "Thai", "English", "Japanese")
        model_name: The Gemini model to use
    """
    logger.info(
        f"Starting text translation to {target_language} with Gemini model: {model_name}"
    )
    prompt = f"""
    You are an expert translator. Translate the following text to {target_language}.
    - Preserve the original meaning and tone
    - Maintain paragraph structure
    - Return ONLY the translated text without any explanations or notes

    Text to translate:
    """

    def translate_single_chunk(chunk):
        """Helper function to translate a single chunk."""
        response = client.models.generate_content(
            model=model_name, contents=prompt + chunk
        )
        return response.text or ""

    try:
        # Check if text needs chunking
        if len(text) > 10000:
            logger.info(
                f"Text length ({len(text)} chars) exceeds 10k, using text-aware chunking"
            )
            chunks = chunk_text_aware(text, max_chunk_size=8000)
            logger.info(f"Split into {len(chunks)} chunks, translating in parallel...")

            # Translate chunks in parallel
            translated_chunks = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all translation tasks
                future_to_index = {
                    executor.submit(translate_single_chunk, chunk): i
                    for i, chunk in enumerate(chunks)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        translated_chunk = future.result()
                        translated_chunks[index] = translated_chunk
                        logger.info(
                            f"Chunk {index + 1}/{len(chunks)} translation completed. Output length: {len(translated_chunk)} characters"
                        )
                    except Exception as e:
                        logger.error(f"Chunk {index + 1} translation failed: {e}")
                        translated_chunks[index] = chunks[index]  # Fallback to original

            # Combine all translated chunks in order
            translated_text = "\n\n".join(translated_chunks)
            logger.info(
                f"All chunks translated successfully. Total output length: {len(translated_text)} characters"
            )
            return translated_text
        else:
            # Text is small enough to process in one go
            logger.info(
                f"Calling Gemini API for translation. Input length: {len(text)} characters, Target: {target_language}"
            )
            response = client.models.generate_content(
                model=model_name, contents=prompt + text
            )
            translated_text = response.text or ""
            logger.info(
                f"Translation completed successfully. Output length: {len(translated_text)} characters"
            )
            return translated_text
    except Exception as e:
        logger.error(f"Translation failed with error: {e}")
        st.error(f"Translation Error: {e}")
        return None


def generate_gemini_tts(text, model_name, voice_name, voice_direction=None):
    """
    Generates audio using the selected Gemini 2.5 model.

    Args:
        text: The text to convert to speech
        model_name: The Gemini model to use
        voice_name: The voice persona to use
        voice_direction: Optional instruction for how the voice should sound
    """
    logger.info(
        f"Starting TTS generation with model: {model_name}, voice: {voice_name}"
    )

    # Default prompt if none provided
    default_prompt = "Read this text in a natural, engaging voice suitable for an audiobook. Speak clearly and at a comfortable pace."

    # Combine text with voice direction
    if voice_direction and voice_direction.strip():
        full_content = f"{voice_direction.strip()}\n\n{text}"
        logger.info(f"Using custom voice direction: {voice_direction[:100]}...")
    else:
        full_content = f"{default_prompt}\n\n{text}"
        logger.info("Using default voice direction prompt")

    try:
        text_length = len(text)
        logger.info(f"Calling Gemini TTS API. Text length: {text_length} characters")
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

        audio_size = len(audio_data)
        logger.info(
            f"TTS generation completed successfully. Audio size: {audio_size} bytes"
        )
        return audio_data

    except Exception as e:
        logger.error(f"TTS generation failed with error: {e}")
        st.error(f"TTS Error: {e}")
        return None


# --- UI LAYOUT ---
st.set_page_config(page_title="Gemini 2.5 Ebook Narrator", layout="wide")

st.sidebar.title("⚙️ Settings")

# 1. Language Selector (placed at top as requested)
target_language = st.sidebar.selectbox(
    "Translate to Language",
    options=list(LANGUAGE_MAP.keys()),
    format_func=lambda x: f"{x} ({LANGUAGE_MAP[x]})",
    index=1,
    help="Select the language to translate the text to.",
)

# 2. Translation Model Selector
translation_model = st.sidebar.selectbox(
    "Translation Model",
    ("gemini-3-flash-preview", "gemini-3-pro-preview"),
    index=0,
    help="Flash is faster & cheaper. Pro produces higher quality translations.",
)

# 3. TTS Model Selector
model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    ("models/gemini-2.5-flash-preview-tts", "models/gemini-2.5-pro-preview-tts"),
    index=0,
    help="Flash is faster & cheaper. Pro has better emotional range.",
)

# 4. Voice Selector
voice_choice = st.sidebar.selectbox(
    "Choose Voice Persona",
    options=list(VOICE_MAP.keys()),
    format_func=lambda x: f"{x} ({VOICE_MAP[x]})",
    index=0,
)

# 5. Voice Direction Input
voice_direction = st.sidebar.text_area(
    "Voice Direction (Optional)",
    placeholder="Describe how you want the voice to sound. Example: 'Speak in a calm, soothing tone perfect for bedtime stories' or 'Read with excitement and enthusiasm for this adventure story.'",
    height=100,
    help="Provide specific instructions for the voice style. If left empty, a default audiobook-appropriate prompt will be used.",
)
st.title(f"📖 Ebook Narrator ({model_choice})")

if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment")
    st.warning("⚠️ GEMINI_API_KEY not found. Please set it in your environment.")

logger.info("Application started - waiting for file upload")
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

# Initialize file tracking in session state
if "current_file_hash" not in st.session_state:
    st.session_state["current_file_hash"] = None

if uploaded_file and api_key:
    # Calculate file hash to detect new file uploads
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()

    # Check if this is a new file
    is_new_file = st.session_state["current_file_hash"] != file_hash

    if is_new_file:
        logger.info(
            f"New file detected: {uploaded_file.name}, clearing previous processing state"
        )
        # Clear previous state when a new file is uploaded
        st.session_state["current_file_hash"] = file_hash
        st.session_state.pop("clean_text", None)
        st.session_state.pop("translated_language", None)

    logger.info(
        f"Processing uploaded file: {uploaded_file.name}, type: {uploaded_file.type}"
    )
    # --- STEP 1: EXTRACT ---
    with st.spinner("Extracting text..."):
        if uploaded_file.type == "application/pdf":
            logger.info("Processing PDF file for text extraction")
            raw_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
        else:
            logger.info("Processing TXT file for text extraction")
            raw_text = uploaded_file.getvalue().decode("utf-8")
            logger.info(
                f"TXT file decoded successfully. Text length: {len(raw_text)} characters"
            )

    if raw_text:
        logger.info(
            f"Text extraction completed successfully. Raw text length: {len(raw_text)} characters"
        )
        # Layout: Left for Text, Right for Audio
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Text Processing")
            st.caption(f"Extracted: {len(raw_text):,} characters")

            with st.expander("Show Raw Text (preview)", expanded=False):
                st.text_area("Raw", raw_text, height=200)

            # Skip cleaning checkbox
            skip_cleaning = st.checkbox(
                "Skip cleaning (use raw text as-is)",
                value=False,
                help="Enable to use the raw extracted text without cleaning. Useful if the text is already clean.",
            )

            # Clean Text Button or Skip button
            if skip_cleaning:
                if st.button("Use Raw Text (Skip Cleaning)"):
                    logger.info("User chose to skip cleaning, using raw text")
                    st.session_state["clean_text"] = raw_text
                    st.session_state["skipped_cleaning"] = True
                    st.success("Using raw text without cleaning!")
                    logger.info("Raw text stored in session state (cleaning skipped)")
                    st.rerun()
            else:
                if st.button("Clean Text (Remove Headers/Footers)"):
                    logger.info("User clicked 'Clean Text' button")
                    st.info(f"Processing {len(raw_text):,} characters...")
                    with st.spinner("Cleaning text with Gemini Flash..."):
                        # We always use Flash for cleaning to save money,
                        # unless you want Pro to do the cleaning too.
                        clean_text = clean_text_with_gemini(raw_text)
                        st.session_state["clean_text"] = clean_text
                        st.session_state["skipped_cleaning"] = False
                        logger.info(
                            "Text cleaning completed and stored in session state"
                        )
                        st.rerun()

        # --- STEP 2: GENERATE ---
        with col2:
            st.subheader("2. Audio Generation")

            if "clean_text" in st.session_state:
                # Show translated language indicator
                text_label = "Cleaned Text"
                if "translated_language" in st.session_state:
                    text_label = f"Text ({st.session_state['translated_language']})"

                st.caption(
                    f"{text_label}: {len(st.session_state['clean_text']):,} characters"
                )
                st.text_area(
                    "Processed Text",
                    st.session_state["clean_text"],
                    height=400,
                    label_visibility="collapsed",
                )

                st.write(f"**Selected Model:** `{model_choice}`")

                # Translate Text Button (moved here since text should be cleaned first)
                if "translated_language" not in st.session_state:
                    if st.button(f"Translate to {target_language}"):
                        text_len = len(st.session_state["clean_text"])
                        logger.info(
                            f"User clicked 'Translate' button for {target_language}"
                        )
                        st.info(
                            f"Translating {text_len:,} characters using {translation_model}..."
                        )
                        with st.spinner(f"Translating text to {target_language}..."):
                            translated_text = translate_text_with_gemini(
                                st.session_state["clean_text"],
                                target_language,
                                translation_model,
                            )
                            if translated_text:
                                st.session_state["clean_text"] = translated_text
                                st.session_state["translated_language"] = (
                                    target_language
                                )
                                logger.info(
                                    "Text translation completed and stored in session state"
                                )
                                st.rerun()

                if st.button("Generate Audio"):
                    logger.info(
                        f"User clicked 'Generate Audio' button with model: {model_choice}, voice: {voice_choice}"
                    )
                    with st.spinner(f"Synthesizing audio using {model_choice}..."):
                        audio_bytes = generate_gemini_tts(
                            st.session_state["clean_text"],
                            model_choice,
                            voice_choice,
                            voice_direction,
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
                            logger.info(
                                "Audio generated successfully and provided to user for download"
                            )
                        else:
                            logger.error(
                                "Audio generation failed - no audio data returned"
                            )
            else:
                st.info("👈 Please clean the text first.")
