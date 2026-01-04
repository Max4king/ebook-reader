"""Gemini AI service for text cleaning, translation, and TTS generation."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google.genai import types

from config import client, logger
from utils.text_processing import chunk_text_aware

logger = logging.getLogger(__name__)


def clean_text_with_gemini(raw_text, model_name="gemini-3-flash-preview"):
    """
    Uses Gemini to remove headers/footers.

    Handles large texts by chunking them into smaller pieces using
    text-aware chunking that respects sentence boundaries and paragraphs.

    We generally default to Flash for this because it's cheap and text-only.
    """
    logger.info(f"Starting text cleaning with Gemini model: {model_name}")
    prompt = """
    You are an expert ebook formatter.
    1. Remove headers, footers, and copyright notices.
    2. IMPORTANT: Keep page numbers in the format [Page X] or (Page X) to maintain structure.
    3. Join hyphenated words split across lines.
    4. Return ONLY the clean text. Do not summarize.

    Text to clean:
    """
    try:
        # Check if text needs chunking
        if len(raw_text) > 15000:
            logger.info(
                f"Text length ({len(raw_text)} chars) exceeds 15k, using text-aware chunking"
            )
            chunks = chunk_text_aware(raw_text, max_chunk_size=10000)
            cleaned_chunks = []

            for i, chunk in enumerate(chunks):
                logger.info(
                    f"Cleaning chunk {i + 1}/{len(chunks)} ({len(chunk)} characters)"
                )
                response = client.models.generate_content(
                    model=model_name, contents=prompt + chunk
                )
                cleaned_chunk = response.text or ""
                cleaned_chunks.append(cleaned_chunk)
                logger.info(
                    f"Chunk {i + 1} cleaning completed. Output length: {len(cleaned_chunk)} characters"
                )

            # Combine all cleaned chunks
            cleaned_text = "\n\n".join(cleaned_chunks)
            logger.info(
                f"All chunks cleaned successfully. Total output length: {len(cleaned_text)} characters"
            )
            return cleaned_text
        else:
            # Text is small enough to process in one go
            text_to_process = raw_text
            input_length = len(text_to_process)
            logger.info(
                f"Calling Gemini API for text cleaning. Input length: {input_length} characters"
            )
            response = client.models.generate_content(
                model=model_name, contents=prompt + text_to_process
            )
            cleaned_text = response.text or ""
            cleaned_length = len(cleaned_text)
            logger.info(
                f"Text cleaning completed successfully. Output length: {cleaned_length} characters"
            )
            return cleaned_text
    except Exception as e:
        logger.error(f"Text cleaning failed with error: {e}")
        st.error(f"Cleaning Error: {e}")
        return raw_text


def translate_text_with_gemini(
    text, target_language, model_name="gemini-3-flash-preview"
):
    """
    Translates text to the target language using Gemini.

    Handles large texts by chunking them into smaller pieces using
    text-aware chunking that respects sentence boundaries and paragraphs.
    Uses parallel processing for faster translation of multiple chunks.

    Args:
        text: The text to translate
        target_language: The language to translate to (e.g., "Thai", "English", "Japanese")
        model_name: The Gemini model to use
    """
    logger.info(
        f"Starting text translation to {target_language} with Gemini model: {model_name}"
    )
    prompt = f"""
    You are an expert translator. Translate the following text to {target_language}.
    - Preserve the original meaning and tone
    - Maintain paragraph structure
    - Return ONLY the translated text without any explanations or notes

    Text to translate:
    """

    def translate_single_chunk(chunk):
        """Helper function to translate a single chunk."""
        response = client.models.generate_content(
            model=model_name, contents=prompt + chunk
        )
        return response.text or ""

    try:
        # Check if text needs chunking
        if len(text) > 10000:
            logger.info(
                f"Text length ({len(text)} chars) exceeds 10k, using text-aware chunking"
            )
            chunks = chunk_text_aware(text, max_chunk_size=8000)
            logger.info(f"Split into {len(chunks)} chunks, translating in parallel...")

            # Translate chunks in parallel
            translated_chunks = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all translation tasks
                future_to_index = {
                    executor.submit(translate_single_chunk, chunk): i
                    for i, chunk in enumerate(chunks)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        translated_chunk = future.result()
                        translated_chunks[index] = translated_chunk
                        logger.info(
                            f"Chunk {index + 1}/{len(chunks)} translation completed. Output length: {len(translated_chunk)} characters"
                        )
                    except Exception as e:
                        logger.error(f"Chunk {index + 1} translation failed: {e}")
                        translated_chunks[index] = chunks[index]  # Fallback to original

            # Combine all translated chunks in order
            translated_text = "\n\n".join(translated_chunks)
            logger.info(
                f"All chunks translated successfully. Total output length: {len(translated_text)} characters"
            )
            return translated_text
        else:
            # Text is small enough to process in one go
            logger.info(
                f"Calling Gemini API for translation. Input length: {len(text)} characters, Target: {target_language}"
            )
            response = client.models.generate_content(
                model=model_name, contents=prompt + text
            )
            translated_text = response.text or ""
            logger.info(
                f"Translation completed successfully. Output length: {len(translated_text)} characters"
            )
            return translated_text
    except Exception as e:
        logger.error(f"Translation failed with error: {e}")
        st.error(f"Translation Error: {e}")
        return None


def generate_gemini_tts(text, model_name, voice_name, voice_direction=None):
    """
    Generates audio using the selected Gemini 2.5 model.

    Args:
        text: The text to convert to speech
        model_name: The Gemini model to use
        voice_name: The voice persona to use
        voice_direction: Optional instruction for how the voice should sound
    """
    logger.info(
        f"Starting TTS generation with model: {model_name}, voice: {voice_name}"
    )

    # Default prompt if none provided
    default_prompt = "Read this text in a natural, engaging voice suitable for an audiobook. Speak clearly and at a comfortable pace."

    # Combine text with voice direction
    if voice_direction and voice_direction.strip():
        full_content = f"{voice_direction.strip()}\n\n{text}"
        logger.info(f"Using custom voice direction: {voice_direction[:100]}...")
    else:
        full_content = f"{default_prompt}\n\n{text}"
        logger.info("Using default voice direction prompt")

    try:
        text_length = len(text)
        logger.info(f"Calling Gemini TTS API. Text length: {text_length} characters")
        response = client.models.generate_content(
            model=model_name,
            contents=full_content,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )

        # Concatenate audio parts
        audio_data = b""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_data += part.inline_data.data

        audio_size = len(audio_data)
        logger.info(
            f"TTS generation completed successfully. Audio size: {audio_size} bytes"
        )
        return audio_data

    except Exception as e:
        logger.error(f"TTS generation failed with error: {e}")
        st.error(f"TTS Error: {e}")
        return None
