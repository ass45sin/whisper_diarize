@echo off
title Project Setup Script - Speaker Diarization Tool

REM --- Section 1: Introduction ---
echo ################################################################################
echo # Welcome to the Project Setup Script for the Speaker Diarization Tool!      #
echo # This script will help you set up the necessary environment.                #
echo ################################################################################
echo.
echo This script will attempt to:
echo   1. Check for Python.
echo   2. Create a Python virtual environment (optional).
echo   3. Activate the virtual environment for this script session.
echo   4. Install required Python packages from requirements.txt (excluding WhisperX).
echo   5. Check for FFmpeg.
echo   6. Guide you on Hugging Face CLI login.
echo   7. Guide you on the manual installation of WhisperX.
echo.
echo Please ensure you have an active internet connection for downloading packages.
echo.
pause
cls

REM --- Section 2: Python Check ---
echo ################################################################################
echo # Step 1: Checking for Python installation...                                #
echo ################################################################################
echo.
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ERROR: Python does not seem to be installed or is not added to your PATH.
    echo.
    echo Please download and install Python 3.8 or newer from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, make sure to check the box that says
    echo "Add Python to PATH" or "Add python.exe to PATH".
    echo.
    echo After installing Python, please re-run this script.
    color
    echo.
    pause
    exit /b 1
) else (
    color 0A
    echo Python is installed! Version:
    python --version
    color
    echo.
)
pause
cls

REM --- Section 3: Create Virtual Environment ---
echo ################################################################################
echo # Step 2: Create Python Virtual Environment                                  #
echo ################################################################################
echo.
echo A virtual environment helps keep project dependencies isolated.
echo It is highly recommended to create one for this project (e.g., in a folder named 'venv').
echo.
:VENV_CHOICE
set /p create_venv="Do you want to create a new virtual environment now? (Y/N): "
echo.

if /i "%create_venv%"=="Y" (
    echo Creating virtual environment in a folder named 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo ERROR: Failed to create the virtual environment.
        echo Please check your Python installation and ensure the 'venv' module is available.
        echo You might need to create it manually: python -m venv venv
        color
        echo.
        pause
        set venv_created=0
    ) else (
        color 0A
        echo Virtual environment 'venv' created successfully!
        color
        set venv_created=1
    )
) else if /i "%create_venv%"=="N" (
    echo OK. Skipping virtual environment creation.
    echo Please ensure you manually create and activate a virtual environment
    echo before running the application or installing dependencies.
    set venv_created=0
) else (
    color 0C
    echo Invalid choice. Please enter Y or N.
    color
    goto VENV_CHOICE
)
echo.
pause
cls

REM --- Section 4: Activate Virtual Environment (for this script session) ---
if %venv_created% equ 1 (
    echo ################################################################################
    echo # Step 3: Activating Virtual Environment for this script session...          #
    echo ################################################################################
    echo.
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
        color 0A
        echo Virtual environment 'venv' activated for this script session.
        color
    ) else (
        color 0C
        echo ERROR: venv\Scripts\activate.bat not found.
        echo Cannot activate the virtual environment automatically for this session.
        echo Please activate it manually in your terminal to install dependencies if needed:
        echo   venv\Scripts\activate.bat
        color
    )
    echo.
    pause
    cls
)

REM --- Section 5: Install Dependencies ---
echo ################################################################################
echo # Step 4: Installing Python Dependencies from requirements.txt               #
echo ################################################################################
echo.
if exist requirements.txt (
    echo Attempting to install packages using pip...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo ERROR: pip install command failed.
        echo.
        echo Possible reasons:
        echo   - Internet connection issue.
        echo   - Errors in the 'requirements.txt' file.
        echo   - Python/pip setup issues.
        echo   - If you skipped virtual environment creation, ensure you are in an active one.
        echo.
        echo Please check the error messages above for more details.
        color
    ) else (
        color 0A
        echo Python dependencies installed successfully!
        color
    )
) else (
    color 0C
    echo ERROR: requirements.txt not found in the current directory.
    echo Cannot install dependencies. Please ensure the file is present.
    color
)
echo.

echo ################################################################################
echo # IMPORTANT: Manual Installation of WhisperX Required                        #
echo ################################################################################
echo.
echo This script does NOT install WhisperX automatically.
echo WhisperX is required for transcription and must be installed manually.
echo Please refer to the 'Project Setup (Manual)' section in README.md
echo for instructions on cloning the WhisperX repository and installing it
echo using 'pip install .' from within its directory.
echo.
pause
cls

REM --- Section 6: FFmpeg Check ---
echo ################################################################################
echo # Step 5: Checking for FFmpeg installation...                                #
echo ################################################################################
echo.
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo WARNING: FFmpeg does not seem to be installed or is not added to your PATH.
    echo.
    echo FFmpeg is essential for audio processing by this tool.
    echo.
    echo Please download and install FFmpeg from:
    echo   https://ffmpeg.org/download.html (official site, often provides source/builds)
    echo   https://github.com/BtbN/FFmpeg-Builds/releases (popular pre-built binaries for Windows)
    echo.
    echo After installation, ensure the directory containing 'ffmpeg.exe' is added
    echo to your system's PATH environment variable.
    echo.
    echo Alternatively, if you use Chocolatey package manager, you can try:
    echo   choco install ffmpeg
    color
) else (
    color 0A
    echo FFmpeg is installed and accessible in your PATH.
    color
)
echo.
pause
cls

REM --- Section 7: Hugging Face Login Guidance ---
echo ################################################################################
echo # Step 6: Hugging Face Login & Model Licenses                                #
echo ################################################################################
echo.
echo IMPORTANT NEXT STEPS (manual):
echo.
echo 1. Log in to Hugging Face using the command line:
echo.
echo    huggingface-cli login
echo.
echo    You will be prompted for a User Access Token. You can create one
echo    with 'read' permissions at: https://huggingface.co/settings/tokens
echo.
echo 2. Accept Model Licenses:
echo    After logging in, you MUST accept the user agreements for the required
echo    models on the Hugging Face website using the SAME ACCOUNT:
echo      - https://huggingface.co/pyannote/speaker-diarization-3.1 (Click "Access repository")
echo      - https://huggingface.co/pyannote/segmentation-3.0 (Click "Access repository")
echo.
echo These steps are crucial for the application to download and use the models.
echo.
pause
cls

REM --- Section 8: Completion Message ---
echo ################################################################################
echo # Setup Script Completed!                                                      #
echo ################################################################################
echo.
echo This script has finished its automated tasks.
echo.
echo REMEMBER:
echo   - Complete the Hugging Face login and accept model licenses if you haven't.
echo   - Manually install WhisperX as per README.md instructions.
echo   - To run the Python application (`diarize_huggingface_cli.py`),
echo     you need to activate the virtual environment in your terminal session first:
echo.
echo       venv\Scripts\activate.bat
echo.
echo   - Then, run the script using:
echo.
echo       python diarize_huggingface_cli.py
echo.
echo Refer to README.md for more details and troubleshooting.
echo.
pause
exit /b 0
