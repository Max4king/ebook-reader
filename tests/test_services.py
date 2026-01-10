"""Unit tests for youtube_service and whisper_service modules."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Import the modules to test
from services.youtube_service import (
    extract_video_id,
    check_existing_download,
    cleanup_intermediate_files,
    download_youtube_video,
    get_video_info,
)
from services.whisper_service import (
    check_existing_transcript,
    transcribe_audio,
)


class TestExtractVideoId(unittest.TestCase):
    """Test cases for extract_video_id function."""

    def test_standard_youtube_url(self):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_short_youtube_url(self):
        """Test extracting video ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_embed_url(self):
        """Test extracting video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_v_url(self):
        """Test extracting video ID from /v/ URL."""
        url = "https://www.youtube.com/v/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_url_with_parameters(self):
        """Test extracting video ID from URL with extra parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&list=PLtest"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_invalid_url(self):
        """Test that invalid URLs return None."""
        url = "https://example.com/page"
        self.assertIsNone(extract_video_id(url))

    def test_empty_url(self):
        """Test that empty string returns None."""
        self.assertIsNone(extract_video_id(""))

    def test_url_with_timestamp(self):
        """Test URL with timestamp parameter."""
        url = "https://www.youtube.com/watch?v=oY0XwMOSzq4&t=120"
        self.assertEqual(extract_video_id(url), "oY0XwMOSzq4")


class TestCheckExistingDownload(unittest.TestCase):
    """Test cases for check_existing_download function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_existing_files(self):
        """Test when no files exist for the video ID."""
        video, audio = check_existing_download("nonexistent123", self.temp_dir)
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_empty_video_id(self):
        """Test with empty video ID."""
        video, audio = check_existing_download("", self.temp_dir)
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_none_video_id(self):
        """Test with None video ID."""
        video, audio = check_existing_download(None, self.temp_dir)  # type: ignore[arg-type]
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_nonexistent_directory(self):
        """Test with non-existent directory."""
        video, audio = check_existing_download("test123", "/nonexistent/path")
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_both_files_exist(self):
        """Test when both video and audio files exist."""
        video_id = "testVideo123"
        video_file = Path(self.temp_dir) / f"Test Video-{video_id}.mp4"
        audio_file = Path(self.temp_dir) / f"Test Video-{video_id}.m4a"
        video_file.touch()
        audio_file.touch()

        video, audio = check_existing_download(video_id, self.temp_dir)
        self.assertIsNotNone(video)
        self.assertIsNotNone(audio)
        assert video is not None  # For type checker
        assert audio is not None
        self.assertTrue(video.endswith(".mp4"))
        self.assertTrue(audio.endswith(".m4a"))

    def test_only_video_exists(self):
        """Test when only video file exists."""
        video_id = "testVideo456"
        video_file = Path(self.temp_dir) / f"Test Video-{video_id}.mp4"
        video_file.touch()

        video, audio = check_existing_download(video_id, self.temp_dir)
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_only_audio_exists(self):
        """Test when only audio file exists."""
        video_id = "testVideo789"
        audio_file = Path(self.temp_dir) / f"Test Video-{video_id}.m4a"
        audio_file.touch()

        video, audio = check_existing_download(video_id, self.temp_dir)
        self.assertIsNone(video)
        self.assertIsNone(audio)

    def test_prefers_non_intermediate_files(self):
        """Test that non-intermediate files are preferred over format-specific files."""
        video_id = "testVideo000"
        # Create intermediate files
        intermediate_video = Path(self.temp_dir) / f"Test Video-{video_id}.f401.mp4"
        intermediate_audio = Path(self.temp_dir) / f"Test Video-{video_id}.f140.m4a"
        intermediate_video.touch()
        intermediate_audio.touch()
        
        # Create final merged files
        final_video = Path(self.temp_dir) / f"Test Video-{video_id}.mp4"
        final_audio = Path(self.temp_dir) / f"Test Video-{video_id}.m4a"
        final_video.touch()
        final_audio.touch()

        video, audio = check_existing_download(video_id, self.temp_dir)
        self.assertIsNotNone(video)
        self.assertIsNotNone(audio)
        assert video is not None  # For type checker
        assert audio is not None
        # Should return the non-intermediate files
        self.assertNotIn(".f401.", video)
        self.assertNotIn(".f140.", audio)


class TestCleanupIntermediateFiles(unittest.TestCase):
    """Test cases for cleanup_intermediate_files function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_removes_intermediate_files(self):
        """Test that intermediate format-specific files are removed."""
        video_id = "testCleanup123"
        intermediate_video = Path(self.temp_dir) / f"Test Video-{video_id}.f401.mp4"
        intermediate_audio = Path(self.temp_dir) / f"Test Video-{video_id}.f140.m4a"
        final_video = Path(self.temp_dir) / f"Test Video-{video_id}.mp4"
        final_audio = Path(self.temp_dir) / f"Test Video-{video_id}.m4a"
        
        for f in [intermediate_video, intermediate_audio, final_video, final_audio]:
            f.touch()

        cleanup_intermediate_files(video_id, self.temp_dir)

        # Intermediate files should be removed
        self.assertFalse(intermediate_video.exists())
        self.assertFalse(intermediate_audio.exists())
        # Final files should remain
        self.assertTrue(final_video.exists())
        self.assertTrue(final_audio.exists())

    def test_nonexistent_directory(self):
        """Test cleanup with non-existent directory doesn't raise error."""
        # Should not raise any exception
        cleanup_intermediate_files("test123", "/nonexistent/path")


class TestCheckExistingTranscript(unittest.TestCase):
    """Test cases for check_existing_transcript function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_existing_transcript(self):
        """Test finding an existing transcript."""
        audio_path = "/path/to/Test Video-testID123.m4a"
        transcript_path = Path(self.temp_dir) / "Test Video-testID123.txt"
        transcript_content = "This is the transcript content."
        transcript_path.write_text(transcript_content)

        text, path = check_existing_transcript(audio_path, self.temp_dir)
        self.assertEqual(text, transcript_content)
        self.assertEqual(path, str(transcript_path))

    def test_no_existing_transcript(self):
        """Test when no transcript exists."""
        audio_path = "/path/to/nonexistent.m4a"
        text, path = check_existing_transcript(audio_path, self.temp_dir)
        self.assertIsNone(text)
        self.assertIsNone(path)

    def test_empty_transcript(self):
        """Test that empty transcripts are not returned."""
        audio_path = "/path/to/Test Video-empty.m4a"
        transcript_path = Path(self.temp_dir) / "Test Video-empty.txt"
        transcript_path.write_text("")

        text, path = check_existing_transcript(audio_path, self.temp_dir)
        self.assertIsNone(text)
        self.assertIsNone(path)

    def test_whitespace_only_transcript(self):
        """Test that whitespace-only transcripts are not returned."""
        audio_path = "/path/to/Test Video-whitespace.m4a"
        transcript_path = Path(self.temp_dir) / "Test Video-whitespace.txt"
        transcript_path.write_text("   \n\t  \n  ")

        text, path = check_existing_transcript(audio_path, self.temp_dir)
        self.assertIsNone(text)
        self.assertIsNone(path)


class TestDownloadYoutubeVideo(unittest.TestCase):
    """Test cases for download_youtube_video function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('services.youtube_service.yt_dlp.YoutubeDL')
    def test_download_new_video(self, mock_ydl_class):
        """Test downloading a new video."""
        # Set up mock
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {'title': 'Test Video', 'id': 'testVid123'}
        mock_ydl.prepare_filename.return_value = os.path.join(self.temp_dir, 'Test Video-testVid123.mp4')
        
        # Create fake output files
        video_path = Path(self.temp_dir) / "Test Video-testVid123.mp4"
        audio_path = Path(self.temp_dir) / "Test Video-testVid123.m4a"
        video_path.touch()
        audio_path.touch()

        result_video, result_audio = download_youtube_video(
            "https://www.youtube.com/watch?v=testVid123",
            output_dir=self.temp_dir
        )

        self.assertIsNotNone(result_video)
        self.assertIsNotNone(result_audio)

    def test_uses_cache_when_available(self):
        """Test that cached files are used when available."""
        video_id = "cachedVid456"
        video_file = Path(self.temp_dir) / f"Cached Video-{video_id}.mp4"
        audio_file = Path(self.temp_dir) / f"Cached Video-{video_id}.m4a"
        video_file.touch()
        audio_file.touch()

        # This should return cached files without calling yt-dlp
        with patch('services.youtube_service.yt_dlp.YoutubeDL') as mock_ydl:
            result_video, result_audio = download_youtube_video(
                f"https://www.youtube.com/watch?v={video_id}",
                output_dir=self.temp_dir
            )
            # yt-dlp should not be called
            mock_ydl.assert_not_called()

        self.assertIsNotNone(result_video)
        self.assertIsNotNone(result_audio)

    def test_progress_callback_called(self):
        """Test that progress callback is called during download."""
        video_id = "progressVid789"
        video_file = Path(self.temp_dir) / f"Progress Video-{video_id}.mp4"
        audio_file = Path(self.temp_dir) / f"Progress Video-{video_id}.m4a"
        video_file.touch()
        audio_file.touch()

        progress_stages = []
        def progress_callback(stage, progress):
            progress_stages.append((stage, progress))

        download_youtube_video(
            f"https://www.youtube.com/watch?v={video_id}",
            output_dir=self.temp_dir,
            progress_callback=progress_callback
        )

        # Should have called callback at least for cache check and done
        self.assertTrue(len(progress_stages) >= 2)
        self.assertEqual(progress_stages[0][0], 'checking_cache')
        self.assertEqual(progress_stages[-1][0], 'done')


class TestTranscribeAudio(unittest.TestCase):
    """Test cases for transcribe_audio function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_nonexistent_audio_file(self):
        """Test that nonexistent audio file returns None."""
        text, path = transcribe_audio("/nonexistent/audio.m4a", output_dir=self.output_dir)
        self.assertIsNone(text)
        self.assertIsNone(path)

    def test_uses_cached_transcript(self):
        """Test that cached transcript is used when available."""
        audio_path = Path(self.temp_dir) / "test_audio.m4a"
        audio_path.touch()
        
        transcript_path = Path(self.output_dir) / "test_audio.txt"
        transcript_content = "Cached transcript content"
        transcript_path.write_text(transcript_content)

        text, path = transcribe_audio(str(audio_path), output_dir=self.output_dir)
        self.assertEqual(text, transcript_content)
        self.assertEqual(path, str(transcript_path))

    def test_progress_callback_for_cached(self):
        """Test progress callback is called for cached transcript."""
        audio_path = Path(self.temp_dir) / "cached_audio.m4a"
        audio_path.touch()
        
        transcript_path = Path(self.output_dir) / "cached_audio.txt"
        transcript_path.write_text("Cached transcript")

        progress_stages = []
        def progress_callback(stage, progress):
            progress_stages.append((stage, progress))

        transcribe_audio(
            str(audio_path),
            output_dir=self.output_dir,
            progress_callback=progress_callback
        )

        self.assertTrue(len(progress_stages) >= 2)
        self.assertEqual(progress_stages[0][0], 'checking_cache')
        self.assertEqual(progress_stages[-1][0], 'done')

    @patch.dict('sys.modules', {'whisperx': MagicMock(), 'torch': MagicMock()})
    def test_transcription_with_mock(self):
        """Test transcription with mocked whisperx."""
        import sys
        
        audio_path = Path(self.temp_dir) / "real_audio.m4a"
        audio_path.touch()

        # Set up mocks
        mock_torch = sys.modules['torch']
        mock_whisperx = sys.modules['whisperx']
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'segments': [
                {'text': 'Hello world'},
                {'text': 'This is a test'},
            ]
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = MagicMock()

        # Need to reload the module with mocks
        import importlib
        import services.whisper_service
        importlib.reload(services.whisper_service)
        
        text, path = services.whisper_service.transcribe_audio(str(audio_path), output_dir=self.output_dir)

        self.assertIsNotNone(text)
        assert text is not None  # For type checker
        self.assertIn("Hello world", text)
        self.assertIn("This is a test", text)


class TestGetVideoInfo(unittest.TestCase):
    """Test cases for get_video_info function."""

    @patch('services.youtube_service.yt_dlp.YoutubeDL')
    def test_get_video_info_success(self, mock_ydl_class):
        """Test successful video info retrieval."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            'title': 'Test Video',
            'duration': 120,
            'uploader': 'Test Channel'
        }

        info = get_video_info("https://www.youtube.com/watch?v=test123")

        self.assertIsNotNone(info)
        assert info is not None  # For type checker
        self.assertEqual(info['title'], 'Test Video')

    @patch('services.youtube_service.yt_dlp.YoutubeDL')
    def test_get_video_info_failure(self, mock_ydl_class):
        """Test video info retrieval failure."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Network error")

        info = get_video_info("https://www.youtube.com/watch?v=invalid")

        self.assertIsNone(info)


class TestIntegration(unittest.TestCase):
    """Integration tests for the workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloads_dir = os.path.join(self.temp_dir, "downloads")
        self.transcripts_dir = os.path.join(self.temp_dir, "transcripts")
        os.makedirs(self.downloads_dir)
        os.makedirs(self.transcripts_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_cache_workflow(self):
        """Test the full workflow with caching."""
        video_id = "integrationTest123"
        
        # Simulate downloaded files
        video_file = Path(self.downloads_dir) / f"Integration Test-{video_id}.mp4"
        audio_file = Path(self.downloads_dir) / f"Integration Test-{video_id}.m4a"
        video_file.touch()
        audio_file.touch()

        # Simulate transcript
        transcript_file = Path(self.transcripts_dir) / f"Integration Test-{video_id}.txt"
        transcript_file.write_text("Integration test transcript content")

        # Test that cache is found for download
        video, audio = check_existing_download(video_id, self.downloads_dir)
        self.assertIsNotNone(video)
        self.assertIsNotNone(audio)

        # Test that cache is found for transcript
        text, path = check_existing_transcript(str(audio_file), self.transcripts_dir)
        self.assertIsNotNone(text)
        self.assertEqual(text, "Integration test transcript content")


if __name__ == "__main__":
    unittest.main()
