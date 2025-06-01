#!/bin/bash

# --- Section 1: Preamble ---
echo "################################################################################"
echo "# Welcome to the Project Setup Script for the Speaker Diarization Tool!      #"
echo "# This script will help you set up the necessary environment for macOS/Linux.#"
echo "################################################################################"
echo ""
echo "This script will attempt to:"
echo "  1. Check for Python 3 and pip3."
echo "  2. Create a Python virtual environment (recommended)."
echo "  3. Activate the virtual environment for this script's operations."
echo "  4. Install required Python packages from requirements.txt (including WhisperX)."
echo "  5. Check for FFmpeg."
echo "  6. Guide you on Hugging Face CLI login and license acceptance."
echo ""
echo "Please ensure you have an active internet connection for downloading packages."
echo "You might be prompted for your password (sudo) for some installations (e.g., FFmpeg on Linux)."
echo ""
read -p "Press Enter to continue..."
clear

# --- Utility function for colored output ---
color_red() {
    echo -e "\033[0;31m$1\033[0m"
}
color_green() {
    echo -e "\033[0;32m$1\033[0m"
}
color_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

# --- Section 2: Python 3 Check ---
echo "################################################################################"
echo "# Step 1: Checking for Python 3 and pip3 installation...                       #"
echo "################################################################################"
echo ""
if ! command -v python3 &> /dev/null; then
    color_red "ERROR: Python 3 does not seem to be installed or is not in your PATH."
    echo ""
    echo "Please install Python 3.8 or newer."
    echo "  - On macOS, you can use Homebrew: brew install python"
    echo "  - On Debian/Ubuntu, you can use apt: sudo apt update && sudo apt install python3 python3-pip"
    echo "  - Otherwise, download from https://www.python.org/downloads/"
    echo ""
    echo "After installing Python 3, please re-run this script."
    exit 1
else
    color_green "Python 3 is installed!"
    echo "Version: $(python3 --version)"
    echo ""
fi

if ! command -v pip3 &> /dev/null; then
    color_red "ERROR: pip3 does not seem to be installed or is not in your PATH."
    echo ""
    echo "pip3 is required to install Python packages."
    echo "  - On Debian/Ubuntu, if you installed python3, pip3 might be included or installable via: sudo apt install python3-pip"
    echo "  - For other systems, ensure your Python 3 installation includes pip."
    echo ""
    echo "After installing pip3, please re-run this script."
    exit 1
else
    color_green "pip3 is installed!"
    echo "Version: $(pip3 --version)"
    echo ""
fi
read -p "Press Enter to continue..."
clear

# --- Section 3: Create Virtual Environment ---
echo "################################################################################"
echo "# Step 2: Create Python Virtual Environment                                  #"
echo "################################################################################"
echo ""
echo "A virtual environment helps keep project dependencies isolated."
echo "It is highly recommended to create one for this project (e.g., in a folder named 'venv')."
echo ""
VENV_DIR="venv"
venv_created_this_session=0

if [ -d "$VENV_DIR" ]; then
    color_yellow "A directory named '$VENV_DIR' already exists."
    read -p "Do you want to use this existing directory for the virtual environment? (Y/n): " use_existing_venv
    if [[ "$use_existing_venv" =~ ^[Nn]$ ]]; then
        echo "Skipping virtual environment creation/activation by user choice."
        echo "Please ensure you manually manage your Python environment."
        VENV_DIR="" # Clear VENV_DIR so later activation steps are skipped for this script
    else
        echo "Using existing '$VENV_DIR' directory."
        # No need to set venv_created_this_session=1 as we are not creating it now
    fi
else
    read -p "Do you want to create a new virtual environment in '$VENV_DIR'? (Y/n): " create_venv
    if [[ "$create_venv" =~ ^[Nn]$ ]]; then
        echo "Skipping virtual environment creation."
        echo "Please ensure you manually create and activate a virtual environment for the project."
        VENV_DIR=""
    else
        echo "Creating virtual environment in '$VENV_DIR'..."
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            color_red "ERROR: Failed to create the virtual environment."
            echo "Please check your Python 3 installation and ensure the 'venv' module is available."
            echo "You might need to create it manually: python3 -m venv $VENV_DIR"
            VENV_DIR=""
        else
            color_green "Virtual environment '$VENV_DIR' created successfully!"
            venv_created_this_session=1
        fi
    fi
fi
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 4: Activate Virtual Environment (for this script session) ---
activated_for_script=0
if [ -n "$VENV_DIR" ]; then # Only attempt if VENV_DIR is set
    echo "################################################################################"
    echo "# Step 3: Activating Virtual Environment for this script session...          #"
    echo "################################################################################"
    echo ""
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        if [ $? -eq 0 ]; then
            color_green "Virtual environment '$VENV_DIR' activated for this script session."
            activated_for_script=1
        else
            color_red "ERROR: Failed to activate virtual environment for this script session."
        fi
    else
        color_red "ERROR: $VENV_DIR/bin/activate not found."
        echo "Cannot activate the virtual environment automatically for this script session."
    fi
    echo ""
    color_yellow "IMPORTANT: After this script finishes, you will need to activate the"
    color_yellow "virtual environment in your own terminal session to run the Python application:"
    color_yellow "  source $VENV_DIR/bin/activate"
    echo ""
else
    color_yellow "Skipping activation for this script as no virtual environment was specified or created."
fi
read -p "Press Enter to continue..."
clear

# --- Section 5: Install Dependencies ---
echo "################################################################################"
echo "# Step 4: Installing Python Dependencies from requirements.txt               #"
echo "# This will include WhisperX for transcription.                              #"
echo "################################################################################"
echo ""
if [ -f "requirements.txt" ]; then
    echo "Attempting to install packages using pip..."
    if [ $activated_for_script -eq 1 ]; then
        pip install -r requirements.txt # Use pip if venv is reliably activated
    else
        pip3 install -r requirements.txt # Use pip3 as a fallback
    fi

    if [ $? -ne 0 ]; then
        color_red "ERROR: 'pip install -r requirements.txt' command failed."
        echo ""
        echo "Possible reasons:"
        echo "  - Internet connection issue."
        echo "  - Errors in the 'requirements.txt' file."
        echo "  - Python/pip setup issues."
        echo "  - If you opted out of virtual environment creation/activation, ensure one is active."
        echo "  - WhisperX installation (from GitHub) might have specific build dependencies."
        echo "    Please check the WhisperX GitHub page for any prerequisites if errors persist."
        echo ""
        echo "Please check the error messages above for more details."
    else
        color_green "Python dependencies (including WhisperX) installed successfully (or already satisfied)!"
    fi
else
    color_red "ERROR: requirements.txt not found in the current directory."
    echo "Cannot install dependencies. Please ensure the file is present."
fi
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 6: FFmpeg Check ---
echo "################################################################################"
echo "# Step 5: Checking for FFmpeg installation...                                #"
echo "################################################################################"
echo ""
if command -v ffmpeg &> /dev/null; then
    color_green "FFmpeg is installed and accessible in your PATH."
    ffmpeg -version | head -n 1 # Display version
else
    color_yellow "WARNING: FFmpeg does not seem to be installed or is not in your PATH."
    echo ""
    echo "FFmpeg is essential for audio processing by this tool."
    echo ""
    OS_TYPE=$(uname -s)
    if [ "$OS_TYPE" == "Darwin" ]; then
        echo "On macOS, you can install FFmpeg using Homebrew:"
        echo "  brew install ffmpeg"
    elif [ "$OS_TYPE" == "Linux" ]; then
        if command -v apt &> /dev/null; then
            echo "On Debian/Ubuntu based systems, you can install FFmpeg using apt:"
            echo "  sudo apt update && sudo apt install ffmpeg"
        elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
            echo "On Fedora/RHEL based systems, you can install FFmpeg using dnf or yum."
            echo "  You may need to enable a repository like RPM Fusion first."
            echo "  Example: sudo dnf install ffmpeg (after enabling repository)"
        else
            echo "Please consult your Linux distribution's documentation for installing FFmpeg."
        fi
    else
        echo "Please install FFmpeg for your operating system and ensure it is in your PATH."
    fi
    echo ""
    echo "After installing FFmpeg, you might need to open a new terminal or re-run this script."
fi
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 7: Hugging Face CLI Login and Model License Acceptance ---
echo "################################################################################"
echo "# Step 6: Hugging Face CLI Login & Model License Acceptance                  #"
echo "################################################################################"
echo ""
echo "This tool requires access to pre-trained models from Hugging Face."
echo "You need to:
  1. Log in to Hugging Face using the CLI.
  2. Accept the user agreements for the required models on the Hugging Face website."
echo ""

echo "Guidance for Hugging Face CLI Login:"
echo "------------------------------------"
echo "Run the following command in your terminal (after this script finishes if you are not already logged in):"
color_yellow "  huggingface-cli login"
echo "You will be prompted for a Hugging Face User Access Token."
echo "Create one with 'read' permissions at: https://huggingface.co/settings/tokens"
echo ""

echo "Guidance for Accepting Model Licenses:"
echo "--------------------------------------"
echo "Using the *same Hugging Face account* you logged in with via the CLI,"
echo "you MUST visit and accept the terms for the following models:"
color_yellow "  1. pyannote/speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1"
color_yellow "  2. pyannote/segmentation-3.0: https://huggingface.co/pyannote/segmentation-3.0"
echo "(Click on 'Access repository' or similar button on the model pages if you haven't already)"
echo ""

if command -v huggingface-cli &> /dev/null; then
    echo "Attempting to check current Hugging Face login status..."
    if huggingface-cli whoami &> /dev/null; then
        color_green "You appear to be logged in to Hugging Face CLI."
        echo "Current user: $(huggingface-cli whoami)"
        color_yellow "Please ensure you have also accepted the model licenses as described above."
    else
        color_yellow "You do not appear to be logged in to Hugging Face CLI."
        color_yellow "Please run 'huggingface-cli login' after this script completes."
    fi
else
    color_yellow "huggingface-cli is not found. It should have been installed with requirements.txt."
    color_yellow "If login issues persist, ensure it's installed and in your PATH (usually handled by venv activation)."
fi
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 8: Post-Installation Instructions ---
echo "################################################################################"
echo "# Setup Script Completed - IMPORTANT Next Steps                              #"
echo "################################################################################"
echo ""
color_green "The automated setup script has finished."
echo ""

if [ -n "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    color_yellow "REMINDER: To run the Python application, you MUST activate the virtual environment"
    color_yellow "in EVERY NEW terminal session you open for this project:"
    echo "  source $VENV_DIR/bin/activate"
else
    color_yellow "REMINDER: If you opted out of virtual environment creation or if it failed,"
    color_yellow "ensure you have a suitable Python environment active before running the application."
fi
echo ""

echo "After activating the virtual environment (if applicable), and ensuring you've completed"
echo "the Hugging Face login and model license steps, you can run the application with:"
color_yellow "  python diarize_huggingface_cli.py"
echo ""

echo "For the main application to find WhisperX correctly, ensure that the virtual environment where"
echo "it was installed (via requirements.txt) is active."
echo ""
echo "Thank you for using the setup script!"
