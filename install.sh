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
echo "  4. Install required Python packages from requirements.txt (includes WhisperX)."
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
        echo ""
        echo "Please check the error messages above for more details."
    else
        color_green "Python dependencies installed successfully (or already satisfied)!"
    fi
else
    color_red "ERROR: requirements.txt not found in the current directory."
    echo "Cannot install dependencies. Please ensure the file is present."
fi
echo ""


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
            echo "On Debian/Ubuntu based Linux, you can install FFmpeg using apt:"
            echo "  sudo apt update && sudo apt install ffmpeg"
        elif command -v dnf &> /dev/null || command -v yum &> /dev/null; then
            echo "On Fedora/RHEL based Linux, you can install FFmpeg using dnf/yum (may require enabling RPM Fusion repository):"
            echo "  sudo dnf install ffmpeg  # or yum install ffmpeg"
        else
            echo "Please install FFmpeg using your system's package manager."
        fi
        echo "Alternatively, download static builds from https://johnvansickle.com/ffmpeg/"
    else
        echo "Please download and install FFmpeg from: https://ffmpeg.org/download.html"
    fi
    echo ""
    echo "After installation, ensure the directory containing 'ffmpeg' is added"
    echo "to your system's PATH environment variable if it's not already."
fi
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 7: Hugging Face Login Guidance ---
echo "################################################################################"
echo "# Step 6: Hugging Face Login & Model Licenses (Manual Steps)                 #"
echo "################################################################################"
echo ""
color_yellow "IMPORTANT NEXT STEPS (to be done manually in your terminal):"
echo ""
echo "1. Activate the virtual environment (if you created one and it's not already active):"
if [ -n "$VENV_DIR" ]; then
    echo "   source $VENV_DIR/bin/activate"
fi
echo ""
echo "2. Log in to Hugging Face using the command line:"
echo ""
echo "   huggingface-cli login"
echo ""
echo "   You will be prompted for a Hugging Face User Access Token. Create one"
echo "   with 'read' permissions at: https://huggingface.co/settings/tokens"
echo ""
echo "3. Accept Model Licenses:"
echo "   Using the SAME Hugging Face account, you MUST accept the user agreements"
echo "   for the required models on the Hugging Face website:"
echo "     - https://huggingface.co/pyannote/speaker-diarization-3.1 (Click 'Access repository')"
echo "     - https://huggingface.co/pyannote/segmentation-3.0 (Click 'Access repository')"
echo ""
echo "These steps are crucial for the application to download and use the models." 
echo "" 
read -p "Press Enter to continue..." 
clear 

# --- Section 7: Download Models for Offline Use (Optional) ---
echo "##########################################################################" 
echo "# Step 7: Download default models for offline use (optional)" 
echo "##########################################################################" 
echo "" 
read -p "Do you want to download the default pyannote and WhisperX models (~7+ GB)? (Y/n): " download_models
if [[ ! "$download_models" =~ ^[Nn]$ ]]; then
    MODEL_DIR="models"
    mkdir -p "$MODEL_DIR"
    git lfs install
    echo "Cloning pyannote/speaker-diarization-3.1..."
    git clone https://huggingface.co/pyannote/speaker-diarization-3.1 "$MODEL_DIR/speaker-diarization-3.1"
    (cd "$MODEL_DIR/speaker-diarization-3.1" && git lfs pull)
    echo "Cloning pyannote/segmentation-3.0..."
    git clone https://huggingface.co/pyannote/segmentation-3.0 "$MODEL_DIR/segmentation-3.0"
    (cd "$MODEL_DIR/segmentation-3.0" && git lfs pull)
    echo "Cloning WhisperX model (faster-whisper-large-v3)..."
    git clone https://huggingface.co/guillaumekln/faster-whisper-large-v3 "$MODEL_DIR/faster-whisper-large-v3"
    (cd "$MODEL_DIR/faster-whisper-large-v3" && git lfs pull)
else
    echo "Skipping model download."
fi
read -p "Press Enter to continue..."
clear

# --- Section 8: Make Script Executable (Guidance) ---
echo "################################################################################"
echo "# Note on Script Execution                                                     #"
echo "################################################################################"
echo ""
echo "If you downloaded this script (install.sh), you might need to make it executable"
echo "before running it next time or if you share it:"
echo "  chmod +x install.sh"
echo ""
read -p "Press Enter to continue..."
clear

# --- Section 9: Completion Message ---
echo "################################################################################"
echo "# Setup Script Completed!                                                      #"
echo "################################################################################"
echo ""
color_green "This script has finished its automated tasks."
echo ""
echo "REMEMBER THE FOLLOWING MANUAL STEPS if you haven't done them yet:"
echo "  - Complete the Hugging Face login ('huggingface-cli login')."
echo "  - Accept the model licenses on the Hugging Face website."
echo ""
if [ -n "$VENV_DIR" ]; then
    echo "To run the Python application ('diarize_huggingface_cli.py'),"
    echo "first activate the virtual environment in your terminal session:"
    echo ""
    color_yellow "    source $VENV_DIR/bin/activate"
    echo ""
else
    echo "If you are using a virtual environment, make sure it's activated"
    echo "before running the Python application."
    echo ""
fi
echo "Then, run the script using:"
echo ""
echo "    python3 diarize_huggingface_cli.py"
echo ""
echo "Refer to README.md for more details and troubleshooting."
echo ""
echo "Exiting setup script."
exit 0
