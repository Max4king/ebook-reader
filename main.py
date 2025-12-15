import argparse
import base64
import sys
from google import genai
from google.genai import types

# 1. SETUP: Paste your API key here
# (Or better, set it as an environment variable: export GOOGLE_API_KEY='your_key')
API_KEY = "AIzaSyCTF1hgun1lsIfGo3A0TqKP__IHLFh4VSA"

def generate_thai_audio(text_to_speak, filename="output_thai.wav"):
    """
    Uses Gemini 2.5 Pro to generate native audio from text.
    """
    client = genai.Client(api_key=API_KEY)

    # 2. PROMPT: Instruct the model to speak in Thai
    # We ask it to act as a native speaker to ensure correct tone/prosody.
    prompt = f"""
    Please read the following Thai text aloud. 
    Speak naturally, clearly, and with a polite tone suitable for a native Thai speaker.
    
    Text to read: "{text_to_speak}"
    """

    print(f"Generating audio for: {text_to_speak}...")

    try:
        # 3. CALL API with AUDIO modality
        # We request 'AUDIO' in response_modalities to get direct speech.
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            # Options often include 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'
                            # 'Aoede' is usually good for formal/polite tones.
                            voice_name="Aoede" 
                        )
                    )
                )
            )
        )

        # 4. SAVE THE AUDIO
        # The audio data comes back as base64 encoded bytes in the response parts
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                audio_bytes = base64.b64decode(part.inline_data.data)
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                print(f"✅ Success! Audio saved to '{filename}'")
                return

        print("⚠️ No audio data found in response.")

    except Exception as e:
        print(f"❌ Error: {e}")

def read_text_file(file_path):
    """
    Read text content from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file '{file_path}': {e}")
        sys.exit(1)

# --- USAGE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Thai audio from text file using Gemini 2.5 Pro"
    )
    parser.add_argument(
        "input_file",
        help="Path to text file containing Thai text to convert to speech"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_thai.wav",
        help="Output audio filename (default: output_thai.wav)"
    )

    args = parser.parse_args()

    # Read text from file
    thai_text = read_text_file(args.input_file)

    if not thai_text:
        print("❌ Error: Input file is empty.")
        sys.exit(1)

    print(f"Read text from '{args.input_file}': {thai_text[:50]}{'...' if len(thai_text) > 50 else ''}")

    generate_thai_audio(thai_text, args.output)
