"""YouTube downloader service using yt-dlp."""

import logging
import os
import re
from pathlib import Path
from typing import Callable

import yt_dlp

from config import DOWNLOADS_DIR

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from URL.

    Args:
        url: YouTube video URL

    Returns:
        Video ID or None if not found
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def check_existing_download(video_id: str, output_dir: str = DOWNLOADS_DIR) -> tuple[str | None, str | None]:
    """
    Check if a video with the given ID has already been downloaded.

    Args:
        video_id: YouTube video ID
        output_dir: Directory where downloads are stored

    Returns:
        Tuple of (video_path, audio_path) or (None, None) if not found
    """
    if not video_id:
        return None, None
    
    # Look for files containing the video ID
    output_path = Path(output_dir)
    if not output_path.exists():
        return None, None
    
    video_path = None
    audio_path = None
    
    # Search for video and audio files - prefer files without format codes (f140, f401, etc.)
    video_candidates = []
    audio_candidates = []
    
    for file in output_path.iterdir():
        if file.is_file() and video_id in file.name:
            # Check if this is a format-specific intermediate file (e.g., .f140.m4a, .f401.mp4)
            # These have format codes before the extension
            name_parts = file.stem.split('.')
            is_intermediate = len(name_parts) > 1 and name_parts[-1].startswith('f') and name_parts[-1][1:].isdigit()
            
            if file.suffix in ('.mp4', '.webm', '.mkv'):
                video_candidates.append((str(file), is_intermediate))
            elif file.suffix in ('.m4a', '.mp3', '.opus', '.wav'):
                audio_candidates.append((str(file), is_intermediate))
    
    # Prefer non-intermediate files (final merged files)
    for path, is_intermediate in video_candidates:
        if not is_intermediate:
            video_path = path
            break
    if not video_path and video_candidates:
        video_path = video_candidates[0][0]  # Fallback to any video file
    
    for path, is_intermediate in audio_candidates:
        if not is_intermediate:
            audio_path = path
            break
    if not audio_path and audio_candidates:
        audio_path = audio_candidates[0][0]  # Fallback to any audio file
    
    # Only return if both files are found
    if video_path and audio_path:
        logger.info(f"Found existing download for video ID {video_id}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Audio: {audio_path}")
        return video_path, audio_path
    
    return None, None


def cleanup_intermediate_files(video_id: str, output_dir: str = DOWNLOADS_DIR) -> None:
    """
    Remove intermediate format-specific files (e.g., .f140.m4a, .f401.mp4).
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory where downloads are stored
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    for file in output_path.iterdir():
        if file.is_file() and video_id in file.name:
            # Check if this is a format-specific intermediate file
            name_parts = file.stem.split('.')
            if len(name_parts) > 1 and name_parts[-1].startswith('f') and name_parts[-1][1:].isdigit():
                logger.info(f"Removing intermediate file: {file.name}")
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove intermediate file {file.name}: {e}")


def download_youtube_video(
    url: str, 
    output_dir: str = DOWNLOADS_DIR,
    progress_callback: Callable[[str, float], None] | None = None,
    force_redownload: bool = False,
) -> tuple[str | None, str | None]:
    """
    Download a YouTube video and extract audio using yt-dlp.
    Checks for existing downloads first to avoid re-downloading.

    Args:
        url: YouTube video URL
        output_dir: Directory to save downloaded files
        progress_callback: Optional callback(stage, progress) for progress updates.
                          Stage is one of: 'checking_cache', 'downloading', 'extracting_audio', 'done'
                          Progress is a float from 0.0 to 1.0
        force_redownload: If True, redownload even if cached files exist

    Returns:
        Tuple of (video_path, audio_path) or (None, None) if failed
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if progress_callback:
        progress_callback('checking_cache', 0.0)
    
    # Check if video is already downloaded (skip if force_redownload)
    video_id = extract_video_id(url)
    if video_id and not force_redownload:
        existing_video, existing_audio = check_existing_download(video_id, output_dir)
        if existing_video and existing_audio:
            logger.info(f"Using cached download for video ID: {video_id}")
            if progress_callback:
                progress_callback('done', 1.0)
            return existing_video, existing_audio
    
    # Delete existing files if force redownload is enabled
    if force_redownload and video_id:
        logger.info(f"Force redownload enabled - deleting cached files for video ID: {video_id}")
        existing_video, existing_audio = check_existing_download(video_id, output_dir)
        if existing_video or existing_audio:
            # Delete video file
            if existing_video and os.path.exists(existing_video):
                try:
                    os.remove(existing_video)
                    logger.info(f"Deleted cached video: {existing_video}")
                except Exception as e:
                    logger.warning(f"Failed to delete cached video: {e}")
            
            # Delete audio file
            if existing_audio and os.path.exists(existing_audio):
                try:
                    os.remove(existing_audio)
                    logger.info(f"Deleted cached audio: {existing_audio}")
                except Exception as e:
                    logger.warning(f"Failed to delete cached audio: {e}")
            
            # Also clean up any intermediate files
            cleanup_intermediate_files(video_id, output_dir)

    if progress_callback:
        progress_callback('downloading', 0.1)

    # Get video title for filename (include video ID for easier identification)
    template = os.path.join(output_dir, "%(title)s-%(id)s.%(ext)s")

    # Download video once and extract audio from it
    logger.info(f"Downloading video from: {url}")
    
    video_path = None
    audio_path = None
    
    # Progress hook for yt-dlp
    def yt_progress_hook(d):
        if progress_callback and d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                # Map download progress to 0.1 - 0.7 range
                pct = 0.1 + (downloaded / total) * 0.6
                progress_callback('downloading', pct)
        elif progress_callback and d['status'] == 'finished':
            progress_callback('extracting_audio', 0.8)
    
    try:
        # Download highest quality video and audio, then merge
        # YouTube's best quality is usually separate streams that need merging
        ydl_opts = {
            # Download best video + best audio and merge (prioritizes quality over format)
            # Falls back to best single file if merging fails
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
            'outtmpl': template,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [yt_progress_hook],
            # Merge into single mp4 file
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            # Keep the video file after extracting audio
            'keepvideo': True,
            # Clean up intermediate files
            'postprocessor_args': {
                'FFmpegExtractAudio': ['-ar', '16000'],  # Optimal for Whisper
            },
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # The filename after merging should be .mp4
            video_filename = ydl.prepare_filename(info)
            # Ensure we're looking for .mp4 file
            video_path = str(Path(video_filename).with_suffix('.mp4'))
            # Audio path will be the same filename with .m4a extension
            audio_path = str(Path(video_filename).with_suffix('.m4a'))
            
            logger.info(f"Video downloaded to: {video_path}")
            logger.info(f"Audio extracted to: {audio_path}")
        
        # Clean up any intermediate format-specific files
        if video_id:
            cleanup_intermediate_files(video_id, output_dir)

        if progress_callback:
            progress_callback('done', 1.0)

        # Verify both files exist
        if video_path and os.path.exists(video_path) and audio_path and os.path.exists(audio_path):
            return video_path, audio_path
        else:
            logger.error(f"Failed to create video or audio file")
            return None, None

    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None, None


def get_title_from_filename(video_path: str) -> str | None:
    """
    Extract video title from the cached video filename.
    Filename format: {title}-{video_id}.{ext}

    Args:
        video_path: Path to the video file

    Returns:
        Video title or None if failed
    """
    try:
        filename = Path(video_path).stem  # Get filename without extension
        # Find the last occurrence of a dash followed by an 11-character video ID
        # Video IDs are always 11 characters: alphanumeric, dash, or underscore
        parts = filename.rsplit('-', 1)
        if len(parts) == 2 and len(parts[1]) == 11:
            title = parts[0]
            return title
        return None
    except Exception as e:
        logger.error(f"Error extracting title from filename: {e}")
        return None


def get_video_info(url: str) -> dict | None:
    """
    Get video information from YouTube URL.

    Args:
        url: YouTube video URL

    Returns:
        Dictionary with video info or None if failed
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return None
