"""YouTube downloader service using yt-dlp."""

import logging
import os
import re
from pathlib import Path

import yt_dlp

from config import DOWNLOADS_DIR, logger

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
    
    # Search for video and audio files
    for file in output_path.iterdir():
        if file.is_file() and video_id in file.name:
            if file.suffix == '.mp4' or file.suffix == '.webm':
                video_path = str(file)
            elif file.suffix == '.mp4' or file.suffix == '.mp3':
                audio_path = str(file)
    
    # Only return if both files are found
    if video_path and audio_path:
        logger.info(f"Found existing download for video ID {video_id}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Audio: {audio_path}")
        return video_path, audio_path
    
    return None, None


def download_youtube_video(url: str, output_dir: str = DOWNLOADS_DIR) -> tuple[str | None, str | None]:
    """
    Download a YouTube video and extract audio using yt-dlp.
    Checks for existing downloads first to avoid re-downloading.

    Args:
        url: YouTube video URL
        output_dir: Directory to save downloaded files

    Returns:
        Tuple of (video_path, audio_path) or (None, None) if failed
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if video is already downloaded
    video_id = extract_video_id(url)
    if video_id:
        existing_video, existing_audio = check_existing_download(video_id, output_dir)
        if existing_video and existing_audio:
            logger.info(f"Using cached download for video ID: {video_id}")
            return existing_video, existing_audio

    # Get video title for filename (include video ID for easier identification)
    template = os.path.join(output_dir, "%(title)s-%(id)s.%(ext)s")

    # Download video once and extract audio from it
    logger.info(f"Downloading video from: {url}")
    
    video_path = None
    audio_path = None
    
    try:
        # Download video with embedded audio
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': template,
            'quiet': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp4',
            }],
            # Keep the video file after extracting audio
            'keepvideo': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_filename = ydl.prepare_filename(info)
            video_path = video_filename
            # Audio path will be the same filename with .mp4 extension
            audio_path = str(Path(video_filename).with_suffix('.mp4'))
            
            logger.info(f"Video downloaded to: {video_path}")
            logger.info(f"Audio extracted to: {audio_path}")

        # Verify both files exist
        if video_path and os.path.exists(video_path) and audio_path and os.path.exists(audio_path):
            return video_path, audio_path
        else:
            logger.error(f"Failed to create video or audio file")
            return None, None

    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None, None


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
