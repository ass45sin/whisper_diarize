# Speaker Diarization Tool

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

This section guides you through setting up the project on your local machine manually. For a more automated approach, see the "Automated Installation" section below.

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

### 1. Project Setup (Manual)

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
    This will install `huggingface_hub` which provides the `huggingface-cli` tool.

4.  **Install FFmpeg:** This is an essential external dependency for audio processing.
    *   **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```
    *   **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
    *   **Windows:** Download FFmpeg builds from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) or [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases). After downloading, you'll need to add the `bin` directory (containing `ffmpeg.exe`) to your system's PATH environment variable.

5.  **Hugging Face Login & Model License Agreements:**
    *   **Login to Hugging Face:**
        ```bash
        huggingface-cli login
        ```
        You'll be prompted to enter a Hugging Face User Access Token. Create one with "read" permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    *   **Accept Model Licenses:** You *must* accept the user agreements for the models on the Hugging Face website using the *same account* you logged in with:
        *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (Click "Access repository")
        *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (Click "Access repository")

## 🚀 Automated Installation (Experimental)

For convenience, platform-specific installation scripts are provided to automate some of the setup steps. These scripts are experimental and aim to simplify the process. It's recommended to understand the manual steps they perform.

### Windows (`install.bat`)

1.  **Download `install.bat`** to the root directory of the cloned project.
2.  **Run the script:** You can either double-click `install.bat` or run it from a Command Prompt:
    ```cmd
    install.bat
    ```
3.  **Follow the prompts:** The script will:
    *   Check for an existing Python installation.
    *   Offer to create a Python virtual environment (in a folder named `venv`).
    *   Attempt to activate the virtual environment for the script's duration.
    *   Install Python dependencies from `requirements.txt`.
    *   Check if FFmpeg is accessible in your system's PATH and provide guidance if not.
    *   Guide you on the manual steps for Hugging Face CLI login and model license acceptance.

### macOS and Linux (`install.sh`)

1.  **Download `install.sh`** to the root directory of the cloned project.
2.  **Make the script executable:** Open your terminal and navigate to the project directory, then run:
    ```bash
    chmod +x install.sh
    ```
3.  **Run the script:**
    ```bash
    ./install.sh
    ```
4.  **Follow the prompts:** The script will:
    *   Check for Python 3 and pip3.
    *   Offer to create a Python virtual environment (in a folder named `venv`).
    *   Attempt to activate the virtual environment for the script's duration.
    *   Install Python dependencies from `requirements.txt`.
    *   Check if FFmpeg is accessible and provide OS-specific installation advice if not (Homebrew for macOS, apt for Debian/Ubuntu, etc.).
    *   Guide you on the manual steps for Hugging Face CLI login and model license acceptance.

### Important Notes for Automated Installation

*   **Review Scripts:** These scripts automate the manual setup steps. You can review the content of `install.bat` or `install.sh` in a text editor to understand the commands they execute.
*   **FFmpeg Installation:** While the scripts check for FFmpeg and provide common installation commands, you might need to perform additional manual steps depending on your specific OS distribution or if you choose a manual FFmpeg installation. Ensuring FFmpeg is correctly added to your system's PATH is crucial.
*   **Hugging Face Authentication:** The scripts will guide you, but the `huggingface-cli login` process and accepting model licenses on the Hugging Face website are interactive steps you must complete carefully using your Hugging Face account.
*   **Virtual Environment Activation:** After the installation script completes, you **must manually activate the virtual environment** in your terminal session before running the main Python application.
    *   Windows (Command Prompt): `venv\Scripts\activate.bat`
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    *   macOS/Linux: `source venv/bin/activate`

## Usage

Run the script from your terminal (ensure your virtual environment is activated):

```bash
python diarize_huggingface_cli.py
```

This will launch a Gradio web interface in your browser.

Refer to the "Help & Documentation" section within the Gradio UI or the sections below for more detailed instructions on using the interface, supported formats, and troubleshooting.

## 📚 Gradio UI Documentation (Mirrored from UI)

This section mirrors the help documentation available directly within the Gradio user interface.

*(The content from the "Help & Documentation" accordion in the Gradio UI is extensive and largely self-contained. For brevity in this README, users are encouraged to refer to the UI itself or the specific sections below for key information like "Common Issues & Solutions" and "System Requirements" which have been updated with consolidated information.)*

## ⚠️ Common Issues & Solutions

This section provides solutions to common problems you might encounter.

### Authentication & Model Access Issues

*   **Error**: `"❌ Not logged in to Hugging Face CLI"` or `"AUTHORIZATION ERROR:"` or `"401 Unauthorized"`
    *   **Solution**:
        1.  Ensure you have run `huggingface-cli login` and entered a valid Hugging Face User Access Token with "read" permissions.
        2.  Crucially, verify you have accepted the license agreements for **both** required models using the *same Hugging Face account* used for `huggingface-cli login`:
            *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (Click "Access repository")
            *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (Click "Access repository")
        3.  If issues persist, try logging in again: `huggingface-cli login`.

### Model Loading & Network Issues

*   **Error**: `"SSL ERROR: There might be a network or proxy issue"`
    *   **Solution**: This often indicates a problem with SSL certificate verification, possibly due to a corporate proxy or outdated certificates. Try:
        ```bash
        pip install --upgrade certifi
        ```
*   **Error**: `"ONNX RUNTIME ERROR: This is a known issue with the model"`
    *   **Solution**: This can be due to incompatibilities with the ONNX model version. Try reinstalling a specific version of `onnxruntime`:
        ```bash
        pip uninstall onnxruntime onnxruntime-gpu
        pip install onnxruntime==1.15.1
        ```
        (You might need to experiment with other `onnxruntime` versions if 1.15.1 doesn't resolve it, depending on model updates.)
*   **Error**: `"MODEL NOT FOUND ERROR: The model file couldn't be downloaded"` or errors containing `"could not download"`
    *   **Solution**:
        1.  Check your internet connection.
        2.  Ensure you have accepted all necessary model licenses on Hugging Face (see "Authentication & Model Access Issues").
        3.  Try clearing the Hugging Face cache and then re-login:
            *   macOS/Linux: `rm -rf ~/.cache/huggingface`
            *   Windows: `del /s /q %USERPROFILE%\.cache\huggingface`
            Then run `huggingface-cli login` again.

### Audio Processing Issues

*   **Error**: `"❌ FFmpeg is not installed or not in PATH."` or `"Error converting audio file"`
    *   **Solution**: FFmpeg is essential for converting audio files to the format required by the diarization pipeline.
        *   Ensure FFmpeg is installed on your system. See "Install FFmpeg" in the Setup section.
        *   Verify that the directory containing `ffmpeg` (or `ffmpeg.exe` on Windows) is included in your system's PATH environment variable.
        *   If the audio file is corrupted, try a different file.

### Transcription Issues

*   **Error**: `"⚠️ No transcription system is installed."` or `"❌ Failed to load transcription model."`
    *   **Solution**: For transcription, you need either WhisperX (recommended) or standard OpenAI Whisper installed.
        *   **WhisperX (Recommended):**
            ```bash
            pip install git+https://github.com/m-bain/whisperx.git
            ```
        *   **Standard Whisper:**
            ```bash
            pip install openai-whisper
            ```
    *   If a Whisper model fails to load (e.g., "Error loading WhisperX model"):
        *   Check you have enough RAM/VRAM for the selected model size (see "Model Sizes" in the UI's Transcription help tab). Try a smaller model (e.g., "base" or "small").
        *   Ensure the installation of Whisper/WhisperX and its dependencies completed without errors.

### General Troubleshooting

*   **Check Console Output**: The script prints detailed status messages, warnings, and errors to the console (terminal). This output is invaluable for diagnosing issues.
*   **Restart Environment**: If you've made changes or installed new packages, try restarting your Python environment (deactivate and reactivate your virtual environment) or even your machine.
*   **Upgrade Packages**: Consider upgrading key packages if you suspect a version incompatibility:
    ```bash
    pip install --upgrade pyannote.audio huggingface_hub torch torchaudio gradio
    ```

## 📦 System Requirements

### Essential Software & Libraries

*   **Python**: 3.8 or newer
*   **FFmpeg**: Required for audio processing and conversion. Must be installed and accessible in your system's PATH.
*   **Python Libraries** (installed via `pip install -r requirements.txt`):
    *   `gradio`: For the web user interface.
    *   `pyannote.audio` (typically version 3.1 or as specified in `requirements.txt`): Core library for speaker diarization.
    *   `torch` (PyTorch, >= 1.12.0 recommended): Deep learning framework used by `pyannote.audio`.
    *   `torchaudio`: Audio library for PyTorch.
    *   `pandas`: For data manipulation.
    *   `huggingface_hub`: Provides `huggingface-cli` for authentication and model downloads, and is used by `pyannote.audio`.

### Optional Dependencies (for Transcription)

*   **WhisperX (Recommended):**
    *   Installation: `pip install git+https://github.com/m-bain/whisperx.git`
    *   Provides faster and often more accurate transcription with better word-level timestamps.
*   **OpenAI Whisper (Fallback):**
    *   Installation: `pip install openai-whisper`
    *   Standard Whisper implementation.

### Hardware Recommendations

*   **CPU**: A modern multi-core processor is recommended.
*   **RAM**:
    *   Minimum: 8GB (especially if not using large transcription models).
    *   Recommended: 16GB or more, particularly when using larger Whisper models for transcription.
*   **GPU (Optional but Highly Recommended for Speed)**:
    *   An NVIDIA GPU with CUDA support (4GB+ VRAM) will significantly speed up both diarization and transcription. The script automatically uses the GPU if PyTorch detects a compatible CUDA environment.
*   **Disk Space**:
    *   At least 10-15GB of free space is recommended for storing downloaded models (which can be several gigabytes each, especially transcription models) and temporary files.

### Internet Connection

*   **Required**: An active internet connection is necessary for:
    *   Initial download of Python packages.
    *   Downloading models from Hugging Face upon first use.
    *   Hugging Face authentication (`huggingface-cli login`).
*   **Not Required for Normal Operation**: Once models are downloaded and cached, the script can run offline for processing, provided no new models need to be fetched.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your features or bug fixes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
