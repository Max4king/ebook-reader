"""Faster-Whisper transcription service."""

import logging
import os
from pathlib import Path
from typing import Callable
from faster_whisper import WhisperModel, BatchedInferencePipeline

# from config import TRANSCRIPTS_DIR

TRANSCRIPTS_DIR = "transcripts"


logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.mmm format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def check_existing_transcript(audio_path: str, output_dir: str = TRANSCRIPTS_DIR) -> tuple[str | None, str | None]:
    """
    Check if a transcript already exists for the given audio file.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory where transcripts are stored

    Returns:
        Tuple of (transcript_text, transcript_file_path) or (None, None) if not found
    """
    audio_name = Path(audio_path).stem
    transcript_path = os.path.join(output_dir, f"{audio_name}.txt")
    
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            if transcript_text.strip():
                logger.info(f"Found existing transcript: {transcript_path}")
                return transcript_text, transcript_path
        except Exception as e:
            logger.warning(f"Error reading existing transcript: {e}")
    
    return None, None


def transcribe_audio(
    audio_path: str,
    model: str = "large-v2",
    output_dir: str = TRANSCRIPTS_DIR,
    compute_type: str = "float16",
    progress_callback: Callable[[str, float], None] | None = None,
    include_timestamps: bool = True,
    include_speaker_labels: bool = False,
    force_retranscribe: bool = False,
) -> tuple[str | None, str | None]:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio_path: Path to audio file
        model: Whisper model to use (default: large-v2)
        output_dir: Directory to save transcript
        compute_type: Compute type for transcription
        progress_callback: Optional callback(stage, progress) for progress updates.
                          Stage is one of: 'checking_cache', 'loading_model', 'loading_audio', 
                          'transcribing', 'saving', 'done'
                          Progress is a float from 0.0 to 1.0
        include_timestamps: Include timestamps for each segment in the transcript
        include_speaker_labels: Include speaker labels in the transcript (requires diarization)
        force_retranscribe: Force re-transcription even if a cached transcript exists.
                           The old transcript will be renamed as a backup.

    Returns:
        Tuple of (transcript_text, transcript_file_path) or (None, None) if failed
    """
    if progress_callback:
        progress_callback('checking_cache', 0.0)
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None, None

    # Check for existing transcript first
    existing_text, existing_path = check_existing_transcript(audio_path, output_dir)
    if existing_text and existing_path and not force_retranscribe:
        logger.info(f"Using cached transcript for: {audio_path}")
        if progress_callback:
            progress_callback('done', 1.0)
        return existing_text, existing_path
    
    # If force_retranscribe is enabled and transcript exists, rename it as backup
    if force_retranscribe and existing_path and os.path.exists(existing_path):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(audio_path).stem
        backup_path = os.path.join(output_dir, f"{audio_name}_backup_{timestamp}.txt")
        os.rename(existing_path, backup_path)
        logger.info(f"Existing transcript backed up to: {backup_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get audio filename without extension for transcript
    audio_name = Path(audio_path).stem
    transcript_path = os.path.join(output_dir, f"{audio_name}.txt")

    try:
        if progress_callback:
            progress_callback('loading_model', 0.1)
        

        logger.info(f"Starting transcription with model: {model}")
        logger.info(f"Loading Whisper model (this may take a while on first run - downloading ~3GB)...")

        # Determine device
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        logger.info(f"Using device: {device}")

        # Initialize faster-whisper model
        whisper_model = WhisperModel(
            model, device=device, compute_type=compute_type
        )
        # Model initialized
        logger.info(f"Whisper model initialized on device: {device}")
        
        # Use batched inference pipeline for better performance
        if device == "cuda":
            batched_model = BatchedInferencePipeline(model=whisper_model)
        else:
            # For CPU, use standard model (batched pipeline requires CUDA)
            batched_model = whisper_model
        
        logger.info(f"Whisper model loaded successfully")
        
        if progress_callback:
            progress_callback('loading_audio', 0.3)

        logger.info(f"Loading and transcribing audio file: {audio_path}")
        
        if progress_callback:
            progress_callback('transcribing', 0.4)
        
        logger.info("Starting transcription (this may take several minutes for longer audio)...")
        
        # Transcribe with faster-whisper
        # Use batched transcription if available, otherwise standard
        if device == "cuda":
            segments, info = batched_model.transcribe(
                audio_path, 
                batch_size=16,  # type: ignore
                word_timestamps=False,
                vad_filter=True
            )
        else:
            segments, info = whisper_model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True
            )
        
        # Convert segments to list for processing
        segments = list(segments)
        
        if progress_callback:
            progress_callback('saving', 0.9)

        # Log detected language
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Get transcript text
        if segments:
            # Format transcript based on user preferences
            transcript_lines = []
            
            for segment in segments:
                line_parts = []
                
                # Add timestamp if requested
                if include_timestamps:
                    start_time = format_timestamp(segment.start)
                    end_time = format_timestamp(segment.end)
                    line_parts.append(f"[{start_time} --> {end_time}]")
                
                # Note: faster-whisper doesn't support speaker diarization by default
                # Speaker labels would require additional integration with pyannote.audio
                if include_speaker_labels:
                    logger.warning("Speaker labels not supported with faster-whisper basic usage")
                
                # Add the text
                line_parts.append(segment.text.strip())
                
                # Combine all parts
                transcript_lines.append(" ".join(line_parts))
            
            transcript_text = "\n".join(transcript_lines)

            # Save transcript to file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            logger.info(f"Transcription saved to: {transcript_path}")
            logger.info(f"Transcript length: {len(transcript_text)} characters")

            if progress_callback:
                progress_callback('done', 1.0)

            return transcript_text, transcript_path
        else:
            logger.error("No transcription segments found")
            return None, None

    except ImportError as e:
        logger.error(
            f"faster-whisper not installed or missing dependency: {e}. Please install it with: pip install faster-whisper"
        )
        return None, None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


# if __name__ == "__main__":
#     # Simple test
#     test_audio = "downloads/Rust Is Easy-CJtvnepMVAU.m4a"
#     transcript, path = transcribe_audio(
#         test_audio,
#         model="small",
#         include_timestamps=True,
#         include_speaker_labels=False,
#     )
#     if transcript:
#         print(f"Transcript saved to: {path}")
#         print(transcript)
#     else:
#         print("Transcription failed.")