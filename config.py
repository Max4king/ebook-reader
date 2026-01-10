"""Configuration and constants for the Ebook Narrator application."""

import os
import logging
import sys
from google import genai

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ebook_reader.log")],
)
logger = logging.getLogger(__name__)

# API Configuration
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    logger.info("API key loaded successfully from environment")
    client: genai.Client = genai.Client(api_key=api_key)
else:
    logger.error("Failed to load GEMINI_API_KEY from environment")
    sys.exit(1)

# Language mappings
LANGUAGE_MAP = {
    "Thai": "ภาษาไทย",
    "English": "English",
    "Japanese": "日本語",
}

# Voice persona mappings
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

# Model options
TTS_MODELS = ("models/gemini-2.5-flash-preview-tts", "models/gemini-2.5-pro-preview-tts")
TRANSLATION_MODELS = ("gemini-3-flash-preview", "gemini-3-pro-preview")

# Whisper model options
WHISPER_MODELS = (
    "large-v2",
    "large-v3",
    "medium",
    "small",
    "base",
    "tiny",
)

# Whisper transcription options
INCLUDE_TIMESTAMPS = True  # Include timestamps in transcript by default
INCLUDE_SPEAKER_LABELS = False  # Include speaker labels in transcript by default (requires diarization)

# Folder paths
DOWNLOADS_DIR = "downloads"
TRANSCRIPTS_DIR = "transcripts"

# Create directories if they don't exist
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
