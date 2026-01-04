# YouTube Video Caching Feature

## Overview
The YouTube downloader service now includes intelligent caching to avoid re-downloading videos that have already been downloaded. This saves bandwidth, time, and storage space.

## How It Works

### 1. Video ID Extraction
When you provide a YouTube URL, the system extracts the unique 11-character video ID from various URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/v/VIDEO_ID`

### 2. Cache Checking
Before downloading, the system searches the `downloads/` directory for files containing the video ID in their names. If both the video file (.mp4 or .webm) and audio file (.m4a or .mp3) are found, the cached versions are used instead.

### 3. File Naming Convention
Downloaded files now include the video ID in their filenames for easy identification:
```
Title-VideoID.mp4
Title-VideoID.m4a
```

Example: `How computer processors run conditions and loops-dQw4w9WgXcQ.mp4`

## Benefits

1. **Faster Processing**: Instantly reuses already downloaded videos
2. **Bandwidth Savings**: Avoids re-downloading the same video multiple times
3. **Storage Management**: Easy to identify and manage downloaded videos by their unique IDs
4. **User Feedback**: The UI now indicates when a cached video is being used

## Usage

Simply use the application as normal. The caching happens automatically:

1. Enter a YouTube URL
2. Click "Download and Transcribe"
3. If the video is already cached, you'll see: "Using cached download (Video ID: XXX)"
4. If it's a new video, it will download and save for future use

## Technical Implementation

### New Functions

**`extract_video_id(url: str) -> str | None`**
- Extracts the YouTube video ID from various URL formats
- Returns the 11-character video ID or None if not found

**`check_existing_download(video_id: str, output_dir: str) -> tuple[str | None, str | None]`**
- Checks if both video and audio files exist for a given video ID
- Returns (video_path, audio_path) if found, or (None, None) if not

**`download_youtube_video(url: str, output_dir: str) -> tuple[str | None, str | None]`**
- Enhanced to check cache first before downloading
- Uses new filename template with video ID included

## Backwards Compatibility

The system supports both old and new filename formats:
- **Old format**: `Title.mp4` (without video ID)
- **New format**: `Title-VideoID.mp4` (with video ID)

Old files without video IDs in their names won't be detected by the cache system and may be re-downloaded. To benefit from caching, you can either:
1. Re-download videos (they will now include the video ID)
2. Manually rename existing files to include the video ID

## Example

```python
from services.youtube_service import download_youtube_video

# First download (fetches from YouTube)
video_path, audio_path = download_youtube_video("https://youtu.be/dQw4w9WgXcQ")
# Output: Video downloaded to: downloads/Never Gonna Give You Up-dQw4w9WgXcQ.mp4

# Second download (uses cache)
video_path, audio_path = download_youtube_video("https://youtu.be/dQw4w9WgXcQ")
# Output: Using cached download for video ID: dQw4w9WgXcQ
```

## Testing

Run the included test script to verify the caching functionality:
```bash
python test_cache.py
```

This will:
- Test video ID extraction from various URL formats
- Check for existing downloads in the downloads directory
- Display cache status for each found video
