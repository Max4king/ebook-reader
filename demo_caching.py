#!/usr/bin/env python3
"""
Quick demo of the YouTube video caching feature.
This shows how the caching works in practice.
"""

from services.youtube_service import extract_video_id, check_existing_download, download_youtube_video

def demo_caching():
    """Demonstrate the caching functionality."""
    print("="*60)
    print("YouTube Video Caching Demo")
    print("="*60)
    
    # Example YouTube URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print(f"\nTest URL: {test_url}")
    
    # Step 1: Extract video ID
    video_id = extract_video_id(test_url)
    print(f"✓ Extracted Video ID: {video_id}")
    if video_id is None:
        print("✗ Failed to extract video ID. Exiting demo.")
        return
    
    # Step 2: Check for existing download
    print(f"\n📂 Checking cache in 'downloads/' directory...")
    existing_video, existing_audio = check_existing_download(video_id)
    
    if existing_video and existing_audio:
        print(f"✓ Found cached files:")
        print(f"  - Video: {existing_video}")
        print(f"  - Audio: {existing_audio}")
        print(f"\n🎉 Cache HIT! No download needed.")
    else:
        print(f"✗ No cached files found.")
        print(f"\n📥 Would download from YouTube if executed.")
    
    print("\n" + "="*60)
    print("Feature Benefits:")
    print("="*60)
    print("• Saves bandwidth by avoiding duplicate downloads")
    print("• Faster processing - instant access to cached videos")
    print("• Easy file management with video ID in filenames")
    print("• Automatic detection - no user intervention needed")
    print("="*60)

if __name__ == "__main__":
    demo_caching()
