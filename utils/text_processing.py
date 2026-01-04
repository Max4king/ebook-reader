"""Text processing utilities for PDF extraction and text chunking."""

import subprocess
import re
import logging

import streamlit as st

from config import logger

logger = logging.getLogger(__name__)


def extract_text_from_pdf_bytes(file_bytes):
    """Extract raw text from PDF using pdftotext."""
    logger.info("Starting PDF text extraction using pdftotext")
    try:
        logger.info(f"Running pdftotext command with {len(file_bytes)} bytes of input")
        result = subprocess.run(
            ["pdftotext", "-", "-"],
            input=file_bytes,
            capture_output=True,
            check=True,
        )
        extracted_length = len(result.stdout)
        logger.info(
            f"PDF text extraction completed successfully. Extracted {extracted_length} characters"
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        st.error(f"Error extracting PDF: {e}")
        return None


def chunk_text_aware(text, max_chunk_size=5000):
    """
    Split text into chunks using sentence boundaries and paragraph breaks.

    This function is text-aware and respects:
    1. Paragraph breaks (newlines)
    2. Sentence boundaries (., !, ?)
    3. Reasonable chunk sizes while avoiding mid-sentence splits

    Args:
        text: The text to chunk
        max_chunk_size: Maximum characters per chunk (soft limit)

    Returns:
        List of text chunks
    """
    logger.info(f"Starting text-aware chunking for {len(text)} characters")

    # First split by paragraphs (double newlines or single newlines)
    paragraphs = re.split(r"\n+", text.strip())

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph itself is too large, split by sentences
        if len(para) > max_chunk_size:
            # Split paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding this sentence would exceed limit, start new chunk
                if (
                    len(current_chunk) + len(sentence) + 1 > max_chunk_size
                    and current_chunk
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        else:
            # Check if adding this paragraph would exceed limit
            if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Text chunked into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i + 1}: {len(chunk)} characters")
    return chunks
