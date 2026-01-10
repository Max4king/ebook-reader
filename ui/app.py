"""Streamlit UI for the Ebook Narrator application."""

import hashlib
import logging
import os
from pathlib import Path

import streamlit as st

from config import (
    LANGUAGE_MAP,
    TTS_MODELS,
    TRANSLATION_MODELS,
    VOICE_MAP,
    WHISPER_MODELS,
    api_key,
    logger,
)
from services.gemini_service import (
    clean_text_with_gemini,
    generate_gemini_tts,
    translate_text_with_gemini,
)
from services.whisper_service import transcribe_audio, check_existing_transcript
from services.youtube_service import download_youtube_video, get_video_info, extract_video_id, check_existing_download, get_title_from_filename
from utils.text_processing import extract_text_from_pdf_bytes


# Progress stage definitions for the checklist
DOWNLOAD_STAGES = {
    'checking_cache': '🔍 Checking cache...',
    'downloading': '📥 Downloading video...',
    'extracting_audio': '🎵 Extracting audio...',
    'done': '✅ Download complete',
}

TRANSCRIBE_STAGES = {
    'checking_cache': '🔍 Checking transcript cache...',
    'loading_model': '🧠 Loading Whisper model...',
    'loading_audio': '🔊 Loading audio file...',
    'transcribing': '📝 Transcribing audio...',
    'saving': '💾 Saving transcript...',
    'done': '✅ Transcription complete',
}


def render_sidebar():
    """Render the sidebar with settings."""
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
        TRANSLATION_MODELS,
        index=0,
        help="Flash is faster & cheaper. Pro produces higher quality translations.",
    )

    # 3. TTS Model Selector
    model_choice = st.sidebar.selectbox(
        "Choose AI Model",
        TTS_MODELS,
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

    return target_language, translation_model, model_choice, voice_choice, voice_direction


def render_text_processing_column(raw_text):
    """Render the left column for text processing."""
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
                logger.info("Text cleaning completed and stored in session state")
                st.rerun()


def render_audio_generation_column(
    model_choice, voice_choice, voice_direction, target_language, translation_model
):
    """Render the right column for audio generation."""
    st.subheader("2. Audio Generation")

    if "clean_text" in st.session_state:
        # Always show cleaned text
        st.caption(f"Cleaned Text: {len(st.session_state['clean_text']):,} characters")
        cleaned_text_area = st.text_area(
            "Cleaned Text",
            st.session_state["clean_text"],
            height=200,
            label_visibility="visible",
        )

        st.write(f"**Selected Model:** `{model_choice}`")

        # Translate Text Button (moved here since text should be cleaned first)
        if st.button(f"Translate to {target_language}"):
            text_len = len(st.session_state["clean_text"])
            logger.info(f"User clicked 'Translate' button for {target_language}")
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
                    st.session_state["translated_text"] = translated_text
                    st.session_state["translated_language"] = target_language
                    logger.info(
                        "Text translation completed and stored in session state"
                    )
                    st.rerun()

        # Show translated text if available
        if "translated_text" in st.session_state:
            st.divider()
            st.caption(
                f"Translated Text ({st.session_state['translated_language']}): {len(st.session_state['translated_text']):,} characters"
            )
            translated_text_area = st.text_area(
                f"Translated Text ({st.session_state['translated_language']})",
                st.session_state["translated_text"],
                height=200,
                label_visibility="visible",
            )

            # Download button for translated text
            st.download_button(
                f"Download Translated Text ({st.session_state['translated_language']})",
                st.session_state["translated_text"],
                f"translated_text_{st.session_state['translated_language'].lower()}.txt",
                "text/plain",
            )

        if st.button("Generate Audio"):
            # Use translated text if available, otherwise use cleaned text
            text_for_audio = st.session_state.get(
                "translated_text", st.session_state["clean_text"]
            )
            audio_source_label = (
                f" ({st.session_state['translated_language']})"
                if "translated_text" in st.session_state
                else " (Cleaned)"
            )
            logger.info(
                f"User clicked 'Generate Audio' button with model: {model_choice}, voice: {voice_choice}"
            )
            with st.spinner(
                f"Synthesizing audio using {model_choice}{audio_source_label}..."
            ):
                audio_bytes = generate_gemini_tts(
                    text_for_audio,
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
                    logger.error("Audio generation failed - no audio data returned")
    else:
        st.info("👈 Please clean the text first.")


def render_progress_checklist(stages: dict, current_stage: str, progress: float) -> None:
    """
    Render a progress checklist showing the current stage of processing.
    
    Args:
        stages: Dictionary mapping stage keys to display labels
        current_stage: The current stage key
        progress: Progress value from 0.0 to 1.0
    """
    stage_keys = list(stages.keys())
    current_idx = stage_keys.index(current_stage) if current_stage in stage_keys else 0
    
    for i, (stage_key, stage_label) in enumerate(stages.items()):
        if i < current_idx:
            # Completed stages
            st.markdown(f"✅ ~~{stage_label.split(' ', 1)[1]}~~")
        elif i == current_idx:
            # Current stage - show with progress
            if stage_key == 'done':
                st.markdown(f"✅ **{stage_label.split(' ', 1)[1]}**")
            else:
                st.markdown(f"⏳ **{stage_label}** ({int(progress * 100)}%)")
        else:
            # Future stages
            st.markdown(f"⬜ {stage_label.split(' ', 1)[1]}")


def render_youtube_downloader_tab():
    """Render the YouTube downloader tab."""
    st.subheader("YouTube Video Downloader & Transcriber")

    # YouTube URL input
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter the URL of the YouTube video you want to download and transcribe.",
    )

    if not youtube_url:
        st.info("👆 Enter a YouTube URL to get started.")
        return

    # Model selection
    col1, col2 = st.columns([1, 1])
    with col1:
        whisper_model = st.selectbox(
            "Whisper Model",
            options=WHISPER_MODELS,
            index=0,
            help="Select the Whisper model for transcription. larger-v2 is the default.",
        )

    with col2:
        compute_type = st.selectbox(
            "Compute Type",
            options=["float16", "float32", "int8"],
            index=0,
            help="Compute type for transcription. float16 is recommended for GPU.",
        )

    # Transcription formatting options
    st.subheader("Transcription Options")
    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        include_timestamps = st.checkbox(
            "Include Timestamps",
            value=False,
            help="Add timestamps for each audio segment in the transcript (e.g., [00:01:23.456 --> 00:01:28.789]).",
        )
    with col4:
        include_speaker_labels = st.checkbox(
            "Include Speaker Labels",
            value=False,
            help="Add speaker labels to the transcript (requires diarization, may increase processing time).",
        )
    with col5:
        force_retranscribe = st.checkbox(
            "Force Re-transcription",
            value=False,
            help="Re-transcribe the audio even if a cached transcript exists. The old transcript will be backed up.",
        )

    # Initialize session state for YouTube
    if "yt_video_path" not in st.session_state:
        st.session_state["yt_video_path"] = None
    if "yt_audio_path" not in st.session_state:
        st.session_state["yt_audio_path"] = None
    if "yt_transcript" not in st.session_state:
        st.session_state["yt_transcript"] = None
    if "yt_transcript_path" not in st.session_state:
        st.session_state["yt_transcript_path"] = None
    if "yt_progress_stage" not in st.session_state:
        st.session_state["yt_progress_stage"] = None
    if "yt_progress_value" not in st.session_state:
        st.session_state["yt_progress_value"] = 0.0

    # Check cache status before showing button
    video_id = extract_video_id(youtube_url)
    existing_video, existing_audio = check_existing_download(video_id) if video_id else (None, None)
    existing_transcript, _ = check_existing_transcript(existing_audio) if existing_audio else (None, None)
    
    # Show cache status
    if video_id:
        col1, col2, col3 = st.columns(3)
        with col1:
            if existing_video and existing_audio:
                st.success("✅ Video cached")
            else:
                st.info("📥 Video not cached")
        with col2:
            if existing_transcript:
                st.success("✅ Transcript cached")
            else:
                st.info("📝 Transcript not cached")
        with col3:
            st.caption(f"Video ID: `{video_id}`")

    # Download and transcribe button
    if st.button("Download and Transcribe"):
        logger.info(f"User clicked download button for URL: {youtube_url}")
        
        # Progress container
        progress_container = st.container()
        status_text = st.empty()
        progress_bar = st.progress(0)
        checklist_container = st.empty()
        
        def update_progress(phase: str, stage: str, progress: float):
            """Update progress display."""
            st.session_state["yt_progress_stage"] = stage
            st.session_state["yt_progress_value"] = progress
            progress_bar.progress(progress)
            
            if phase == 'download':
                status_text.markdown(f"**{DOWNLOAD_STAGES.get(stage, stage)}**")
            else:
                status_text.markdown(f"**{TRANSCRIBE_STAGES.get(stage, stage)}**")
        
        # Download phase
        update_progress('download', 'checking_cache', 0.0)
        
        if existing_video and existing_audio:
            status_text.info(f"Using cached download (Video ID: {video_id})")
            video_path, audio_path = existing_video, existing_audio
            update_progress('download', 'done', 0.3)
        else:
            status_text.info(f"Downloading video from: {youtube_url}")
            
            # Download with progress callback
            def download_callback(stage, prog):
                # Map download progress to 0-30%
                mapped_prog = prog * 0.3
                update_progress('download', stage, mapped_prog)
            
            video_path, audio_path = download_youtube_video(youtube_url, progress_callback=download_callback)

        if video_path and audio_path:
            st.session_state["yt_video_path"] = video_path
            st.session_state["yt_audio_path"] = audio_path
            
            # Show video info - extract title from cached filename to avoid network request
            title = get_title_from_filename(video_path)
            if title:
                st.caption(f"Title: {title}")

            # Transcription phase
            logger.info(f"Starting transcription with model: {whisper_model}")
            
            def transcribe_callback(stage, prog):
                # Map transcription progress to 30-100%
                mapped_prog = 0.3 + (prog * 0.7)
                update_progress('transcribe', stage, mapped_prog)
            
            transcript, transcript_path = transcribe_audio(
                audio_path, whisper_model, compute_type=compute_type,
                progress_callback=transcribe_callback,
                include_timestamps=include_timestamps,
                force_retranscribe=force_retranscribe,
                include_speaker_labels=include_speaker_labels
            )

            if transcript:
                st.session_state["yt_transcript"] = transcript
                st.session_state["yt_transcript_path"] = transcript_path
                progress_bar.progress(1.0)
                status_text.success("✅ Transcription completed!")
                st.rerun()
            else:
                st.error("Transcription failed. Check the logs for details.")
        else:
            st.error("Failed to download video. Please check the URL and try again.")

    # Show results
    if st.session_state.get("yt_transcript"):
        st.divider()
        st.subheader("Transcript")

        # Show transcript stats
        transcript_len = len(st.session_state["yt_transcript"])
        word_count = len(st.session_state["yt_transcript"].split())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", f"{transcript_len:,}")
        with col2:
            st.metric("Words", f"{word_count:,}")
        with col3:
            st.metric("Model", whisper_model)

        # Show transcript in expandable section
        with st.expander("View Transcript", expanded=True):
            st.text_area(
                "Transcript",
                st.session_state["yt_transcript"],
                height=300,
                label_visibility="collapsed",
            )

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Transcript (TXT)",
                st.session_state["yt_transcript"],
                "transcript.txt",
                "text/plain",
            )
        with col2:
            if st.session_state.get("yt_audio_path") and os.path.exists(
                st.session_state["yt_audio_path"]
            ):
                with open(st.session_state["yt_audio_path"], "rb") as f:
                    audio_bytes = f.read()
                st.download_button(
                    "Download Audio (M4A)",
                    audio_bytes,
                    os.path.basename(st.session_state["yt_audio_path"]),
                    "audio/mp4",
                )


def main():
    """Main function to run the Streamlit app."""
    # --- UI LAYOUT ---
    st.set_page_config(page_title="Gemini 2.5 Ebook Narrator", layout="wide")

    # Render sidebar and get settings
    target_language, translation_model, model_choice, voice_choice, voice_direction = (
        render_sidebar()
    )

    st.title(f"📖 Ebook Narrator & Transcriber")

    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment")
        st.warning("⚠️ GEMINI_API_KEY not found. Please set it in your environment.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["📖 Ebook Narrator", "🎥 YouTube Downloader"])

    # Tab 1: Ebook Narrator
    with tab1:
        st.subheader(f"({model_choice})")
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
                st.session_state.pop("translated_text", None)
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
                    render_text_processing_column(raw_text)

                # --- STEP 2: GENERATE ---
                with col2:
                    render_audio_generation_column(
                        model_choice, voice_choice, voice_direction, target_language, translation_model
                    )

    # Tab 2: YouTube Downloader
    with tab2:
        render_youtube_downloader_tab()


if __name__ == "__main__":
    main()
