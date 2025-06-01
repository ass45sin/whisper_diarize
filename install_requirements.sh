#!/bin/bash

# Function to install packages with retries and fallbacks
install_requirements() {
    local requirements_file="$1"
    local env_type="$2"  # "conda" or "venv"
    local max_retries=3
    local retry_count=0

    echo "📦 Installing requirements..."
    # First update pip and install build dependencies
    pip install --upgrade pip
    pip install --upgrade setuptools wheel build

    # Try to install scipy with conda first if using conda
    if [ "$env_type" = "conda" ] && grep -i "scipy" "$requirements_file" &> /dev/null; then
        echo "📦 Installing scipy with conda..."
        conda install -y scipy numpy
    fi

    # Read requirements and install them one by one with retries
    while IFS= read -r requirement || [ -n "$requirement" ]; do
        if [ -n "$requirement" ] && [[ ! "$requirement" =~ ^[[:space:]]*# ]]; then
            # Skip scipy and numpy as we installed them with conda
            if [ "$env_type" = "conda" ] && ([[ "$requirement" == scipy* ]] || [[ "$requirement" == numpy* ]]); then
                continue
            fi

            retry_count=0
            while [ $retry_count -lt $max_retries ]; do
                echo "Installing: $requirement"
                
                # Try different installation methods based on package
                case "$requirement" in
                    "sentencepiece"*)
                        # For sentencepiece, install from conda-forge
                        if [ "$env_type" = "conda" ]; then
                            if conda install -y -c conda-forge "$requirement"; then
                                break
                            fi
                        fi
                        ;;
                    "av"*)
                        # For av package, try conda-forge first
                        if [ "$env_type" = "conda" ]; then
                            if conda install -y -c conda-forge "av"; then
                                break
                            fi
                        fi
                        ;;
                esac

                # If specific handling didn't work or wasn't applicable, try pip
                if pip install "$requirement"; then
                    break
                fi

                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    echo "⚠️  Retrying installation of $requirement (attempt $((retry_count + 1))/$max_retries)"
                    sleep 2
                else
                    echo "⚠️  Failed to install: $requirement after $max_retries attempts"
                fi
            done
        fi
    done < "$requirements_file"

    touch .requirements_installed
}
