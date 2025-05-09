# Speaker Diarization CLI Tool

This project provides a command-line interface and Gradio web UI for speaker diarization using `pyannote.audio` (version 3.1).
It leverages Hugging Face for model access and authentication, requiring users to log in via `huggingface-cli login`.

The tool can optionally perform speech transcription using WhisperX (recommended) or standard Whisper if installed.

## Features

- Speaker diarization of audio files (WAV, MP3, M4A, FLAC, etc.)
- Transcription of speech segments (optional, requires WhisperX or Whisper)
- Batch processing of multiple audio files from a folder
- Audio trimming capabilities
- Detection of number of speakers (or manual specification)
- Export results as text and structured JSON
- Gradio web interface for ease of use
- Progress tracking for long operations

## Setup

1.  **Clone the repository (if applicable).**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install FFmpeg:** This is required for audio processing. Instructions vary by OS.
    - On macOS (using Homebrew): `brew install ffmpeg`
    - On Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
4.  **Hugging Face Login:**
    ```bash
    huggingface-cli login
    ```
    You'll need a Hugging Face account and a User Access Token (read permission is sufficient). Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
5.  **Accept Model Licenses:**
    You must accept the user agreements for the models on the Hugging Face website:
    - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Usage

Run the script from your terminal:

```bash
python diarize_huggingface_cli.py
```

This will launch a Gradio web interface in your browser.

Refer to the "Help & Documentation" section within the Gradio UI for more detailed instructions on using the interface, supported formats, and troubleshooting. 