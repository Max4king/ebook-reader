"""Whisper-X transcription service."""

import logging
import os
from pathlib import Path

from config import TRANSCRIPTS_DIR, logger

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model: str = "large-v2",
    output_dir: str = TRANSCRIPTS_DIR,
    compute_type: str = "float16",
) -> tuple[str | None, str | None]:
    """
    Transcribe audio using whisper-x.

    Args:
        audio_path: Path to audio file
        model: Whisper model to use (default: large-v2)
        output_dir: Directory to save transcript
        compute_type: Compute type for transcription

    Returns:
        Tuple of (transcript_text, transcript_file_path) or (None, None) if failed
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None, None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get audio filename without extension for transcript
    audio_name = Path(audio_path).stem
    transcript_path = os.path.join(output_dir, f"{audio_name}.txt")

    try:
        import torch
        import whisperx

        # Add safe globals for PyTorch 2.6+ compatibility with Pyannote VAD models
        try:
            from omegaconf import DictConfig, ListConfig
            torch.serialization.add_safe_globals([DictConfig, ListConfig])
        except ImportError:
            pass  # OmegaConf not installed, no need to add safe globals

        logger.info(f"Starting transcription with model: {model}")

        # Initialize whisperx model
        whisper_model = whisperx.load_model(
            model, compute_type=compute_type, device="cuda"
        )

        # Load and transcribe audio
        logger.info(f"Loading audio file: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        result = whisper_model.transcribe(audio, batch_size=16)

        # Get transcript text
        if "segments" in result and result["segments"]:
            transcript_text = "\n".join(
                segment["text"] for segment in result["segments"]
            )

            # Save transcript to file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            logger.info(f"Transcription saved to: {transcript_path}")
            logger.info(f"Transcript length: {len(transcript_text)} characters")

            return transcript_text, transcript_path
        else:
            logger.error("No transcription segments found")
            return None, None

    except ImportError:
        logger.error(
            "whisperx not installed. Please install it with: pip install git+https://github.com/m-bain/whisperx.git"
        )
        return None, None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None, None
