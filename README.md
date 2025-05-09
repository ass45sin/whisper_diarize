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

This section guides you through setting up the project on your local machine.

### 0. Prerequisites: Python Installation

Ensure you have Python 3.8 or newer installed.

*   **General Recommendation:** Download the official Python installer from [python.org](https://www.python.org/downloads/).

*   **macOS:**
    *   Using Homebrew (recommended):
        ```bash
        brew install python
        ```
    *   Alternatively, use the installer from [python.org](https://www.python.org/downloads/macos/).

*   **Linux (Ubuntu/Debian based):**
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv
    ```
    For other distributions, please refer to their respective package managers and documentation.

*   **Windows:**
    *   Download and run the installer from [python.org](https://www.python.org/downloads/windows/). **Important:** Check the box "Add Python to PATH" during installation.
    *   Alternatively, Python can be installed via the Microsoft Store or using winget:
        ```bash
        winget install Python.Python.3
        ```

Verify your installation by opening a terminal or command prompt and typing:
```bash
python3 --version  # or python --version on Windows if python3 isn't aliased
pip3 --version     # or pip --version
```

### 1. Project Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/ass45sin/whisper_diarize.git
    cd whisper_diarize
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects or your global Python installation.

    *   **Create the virtual environment** (e.g., named `venv`):
        ```bash
        python3 -m venv venv
        ```
        (On Windows, you might use `python` instead of `python3` if `python3` is not recognized: `python -m venv venv`)

    *   **Activate the virtual environment:**
        *   On macOS and Linux:
            ```bash
            source venv/bin/activate
            ```
        *   On Windows (Command Prompt):
            ```bash
            venv\\Scripts\\activate.bat
            ```
        *   On Windows (PowerShell):
            ```bash
            .\\venv\\Scripts\\Activate.ps1
            ```
            (If you get an error about script execution policy in PowerShell, you might need to run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` and then try activating again.)

    Your terminal prompt should change to indicate that the virtual environment is active (e.g., `(venv) your-prompt$`).

3.  **Install dependencies:**
    With your virtual environment active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install FFmpeg:** This is required for audio processing. Instructions vary by OS.
    - On macOS (using Homebrew): `brew install ffmpeg`
    - On Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
5.  **Hugging Face Login:**
    ```bash
    huggingface-cli login
    ```
    You'll need a Hugging Face account and a User Access Token (read permission is sufficient). Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
6.  **Accept Model Licenses:**
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

## 📚 Gradio UI Documentation

This section mirrors the help documentation available directly within the Gradio user interface.

### 🚀 Getting Started Guide

1.  **Login to Hugging Face**:
    ```
    huggingface-cli login
    ```
    Enter your token when prompted (create one at https://huggingface.co/settings/tokens)
2.  **Accept Model Licenses**:
    *   Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   Click "Access repository" on both pages
3.  **Upload Audio**:
    *   Choose an audio file in any common format
    *   Adjust settings as needed
    *   Click "Run Diarization"
4.  **View Results**:
    *   See speaker segmentation with timestamps
    *   Results are saved to the specified output folder

### 📋 Supported Audio Formats

This tool supports all common audio formats through FFmpeg conversion:

| Format | Description                                         |
| :----- | :-------------------------------------------------- |
| WAV    | Waveform Audio File (best quality/compatibility)    |
| MP3    | MPEG Audio Layer III (common compressed format)     |
| M4A    | MPEG-4 Audio (Apple format, commonly from recordings) |
| FLAC   | Free Lossless Audio Codec (high quality compressed) |
| OGG    | Ogg Vorbis (free, open compressed format)           |
| AAC    | Advanced Audio Coding (common compressed format)    |

Any other format supported by FFmpeg should also work. Files are automatically
converted to 16kHz mono WAV format for processing.

### 🔧 Key Features

#### Basic Features

*   **Speaker Diarization**: Identify who spoke when
*   **Multiple File Formats**: Process WAV, MP3, M4A, FLAC and more
*   **Speaker Statistics**: See speaking time and percentages for each speaker

#### Advanced Features

*   **Batch Processing**: Process multiple audio files at once
*   **Audio Trimming**: Process only a specific part of an audio file
*   **Transcription**: Convert speech to text using OpenAI Whisper
*   **Structured Output**: Export results as JSON for further processing
*   **GPU Acceleration**: Automatically uses GPU if available
*   **Progress Tracking**: See real-time progress for long processing tasks

### 🎯 Speech Transcription

**Current Status** (this will reflect your local environment when you run the script):
*   WhisperX: Available ✅ (Recommended) / Not Available ❌
*   Whisper: Available ✅ (Fallback) / Not Available ❌

#### Installation

If transcription is not available, install WhisperX (recommended):
```
pip install git+https://github.com/m-bain/whisperx.git
```

Or standard Whisper:
```
pip install openai-whisper
```

#### WhisperX Advantages

*   Better integration with speaker diarization
*   More accurate word-level timestamps
*   Voice activity detection (VAD) for better noise handling
*   Faster processing than standard Whisper

#### Model Sizes

| Size     | Memory | Accuracy | Speed   | Use Case        |
| :------- | :----- | :------- | :------ | :-------------- |
| tiny     | ~1GB   | Low      | Fast    | Quick testing   |
| base     | ~1GB   | Basic    | Fast    | General use     |
| small    | ~2GB   | Good     | Medium  | Better accuracy |
| medium   | ~5GB   | High     | Slow    | High accuracy   |
| large-v3 | ~10GB  | Highest  | Slowest | Best results    |

Specifying a language can improve transcription accuracy significantly.

### 📊 Structured JSON Output

When enabling "Export results as JSON", you get a structured file with:

```json
{
  "file_info": {
    "filename": "recording.mp3",
    "path": "/path/to/recording.mp3",
    "trimmed": false
  },
  "segments": [
    {
      "start": 0.5,
      "end": 10.2,
      "duration": 9.7,
      "speaker": "SPEAKER_01",
      "text": "This is the transcribed text if available"
    }
    // ...
  ],
  "speakers": {
    "SPEAKER_01": {
      "talk_time": 120.5,
      "percentage": 45.3,
      "segments": [0, 2, 5] // indices of segments spoken by this speaker
    }
    // ...
  }
}
```

This format is ideal for:

*   Further data analysis
*   Integration with other tools
*   Building custom visualizations
*   Training machine learning models

### ⚠️ Common Issues & Solutions

#### Authentication Issues

*   **Error**: "Not logged in to Hugging Face CLI"
    **Solution**: Run `huggingface-cli login` and enter your token
*   **Error**: "401 Unauthorized"
    **Solution**: Make sure you've accepted the model licenses on the Hugging Face website

#### Audio Conversion Problems

*   **Error**: "Failed to convert audio file"
    **Solution**: Install FFmpeg or check if your audio file is corrupted

#### Model Loading Failures

*   **Error**: "ONNX Runtime Error"
    **Solution**: Try `pip uninstall onnxruntime onnxruntime-gpu` then `pip install onnxruntime==1.15.1`
*   **Error**: "SSL Error"
    **Solution**: Run `pip install --upgrade certifi`

#### Transcription Problems

*   **Error**: "Failed to load Whisper model"
    **Solution**: Check if you have enough memory for the selected model size, try a smaller model

For further help, check the console output for detailed error messages.

### 📦 System Requirements

#### Essential Dependencies

*   **Python**: 3.8 or newer
*   **PyTorch**: 1.12.0 or newer
*   **FFmpeg**: For audio conversion
*   **pyannote.audio**: For diarization
*   **Hugging Face CLI**: For authentication

#### Optional Dependencies

*   **OpenAI Whisper / WhisperX**: For speech transcription
*   **CUDA**: For GPU acceleration (significantly speeds up processing)

#### Hardware Recommendations

*   **CPU**: Modern multi-core processor
*   **RAM**: 8GB minimum, 16GB+ recommended
*   **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster processing)
*   **Disk Space**: At least 10GB free space for models and temporary files

#### Internet Connection

Required for initial model download and authentication. 