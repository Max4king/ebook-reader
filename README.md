# E Book Reader Web Application

A Streamlit-based web application for converting ebooks to audiobooks using AI-powered text-to-speech, with YouTube video transcription capabilities.

## Features

- **EBook Narration**: Upload PDF/TXT files, clean text, translate to multiple languages, and generate audiobooks using Gemini TTS
- **YouTube Integration**: Download videos, extract audio, and transcribe using Whisper
- **Multi-language Support**: Thai, English, Japanese with configurable voice personas
- **Caching**: Efficient caching for downloads and transcriptions
- **Progress Tracking**: Real-time progress indicators for long-running operations

## Docker Setup

### Prerequisites

- Docker installed on your system
- A Google Gemini API key

### Running with Docker

1. **Create a `.env` file** with your API key:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Or copy the example file:

```bash
cp .env-example .env
# Edit .env and add your API key
```

2. **Build the Docker image**:

```bash
docker build -t ebook-reader .
```

3. **Run the container**:

```bash
docker run -d -p 8501:8501 --env-file .env -v $(pwd)/downloads:/app/downloads -v $(pwd)/transcripts:/app/transcripts --name ebook-reader ebook-reader
```

Or use Docker Compose (recommended):

```bash
docker compose up -d
```

4. **Access the application**:

Open your browser and navigate to `http://localhost:8501`

### Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
services:
  ebook-reader:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./downloads:/app/downloads
      - ./transcripts:/app/transcripts
    restart: unless-stopped
```

Then run:

```bash
docker compose up -d
```

## GitHub Actions

This project uses GitHub Actions to automatically build and push Docker images to GitHub Container Registry (GHCR).

The workflow triggers on:
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

Built images are tagged with:
- Branch name
- Commit SHA
- `latest` for main branch builds

### Pulling the Image

Once built, you can pull the image from GHCR:

```bash
docker pull ghcr.io/YOUR_USERNAME/ebook-reader:latest
```

Replace `YOUR_USERNAME` with your GitHub username.

## Local Development

### Prerequisites

- Python 3.13+
- UV package manager

### Setup

1. Install dependencies:

```bash
uv sync
```

2. Create a `.env` file with your API key

3. Run the application:

```bash
uv run streamlit run main.py
```

## License

