"""Test script to verify YouTube video caching functionality."""

from services.youtube_service import extract_video_id, check_existing_download

# Test the extract_video_id function
test_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/v/dQw4w9WgXcQ",
]

print("Testing video ID extraction:")
print("-" * 50)
for url in test_urls:
    video_id = extract_video_id(url)
    print(f"URL: {url}")
    print(f"Video ID: {video_id}\n")

# Test checking for existing downloads
print("\nTesting existing download check:")
print("-" * 50)
# Look for any videos that might already be in the downloads directory
import os
from pathlib import Path

downloads_dir = "downloads"
if os.path.exists(downloads_dir):
    for file in Path(downloads_dir).iterdir():
        if file.suffix in ['.mp4', '.webm', '.m4a', '.mp3']:
            print(f"Found file: {file.name}")
            # Try to extract video ID from filename
            # The new format will be: Title-VideoID.ext
            parts = file.stem.split('-')
            if len(parts) >= 2:
                potential_id = parts[-1]
                print(f"  Potential Video ID: {potential_id}")
                video, audio = check_existing_download(potential_id, downloads_dir)
                if video and audio:
                    print(f"  ✓ Cache hit!")
                    print(f"    Video: {Path(video).name}")
                    print(f"    Audio: {Path(audio).name}")
                else:
                    print(f"  ✗ Not complete (missing video or audio)")
            print()
