#!/bin/bash

set -e

echo "________________________________________________"
echo "Stage 1: Checking for pyenv and installing if not present"
echo "________________________________________________"

# Check for pyenv and install if not present
if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found, installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    echo "Cloning pyenv repository..."
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
else
    echo "pyenv is already installed"
fi

echo "________________________________________________"
echo "Stage 2: Configuring pyenv in shell"
echo "________________________________________________"

# Detect default shell and add pyenv to the shell configuration file if not already present
DEFAULT_SHELL=$(basename "$SHELL")
if [ "$DEFAULT_SHELL" = "zsh" ]; then
    CONFIG_FILE="$HOME/.zshrc"
else
    CONFIG_FILE="$HOME/.bashrc"
fi

if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' "$CONFIG_FILE"; then
    echo "Adding pyenv configuration to $CONFIG_FILE"
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$CONFIG_FILE"
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$CONFIG_FILE"
    echo 'eval "$(pyenv init --path)"' >> "$CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "pyenv configuration already present in $CONFIG_FILE"
fi

echo "________________________________________________"
echo "Stage 3: Updating pyenv and installing Python 3.12"
echo "________________________________________________"

cd ~/.pyenv && git pull && cd -
echo "Installing Python 3.12 with pyenv"
pyenv install -f 3.12

echo "________________________________________________"
echo "Stage 4: Installing additional dependencies"
echo "________________________________________________"

sudo apt-get update
sudo apt-get install -y libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig redis-server

echo "Enabling and starting Redis server"
sudo systemctl enable redis-server
sudo systemctl start redis

echo "________________________________________________"
echo "Stage 5: Detecting CUDA and updating requirements.txt"
echo "________________________________________________"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    echo "CUDA detected, version: $CUDA_VERSION"
    case $CUDA_VERSION in
        12.1*) CUDA_TAG="cu121" ;;
        12.2*) CUDA_TAG="cu122" ;;
        12.3*) CUDA_TAG="cu123" ;;
        12.4*) CUDA_TAG="cu124" ;;
        *) CUDA_TAG="" ;;
    esac

    if [ -n "$CUDA_TAG" ]; then
        echo "Updating requirements.txt for CUDA version $CUDA_TAG"
        sed -i 's/faiss-cpu/faiss/' requirements.txt
        sed -i 's@llama-cpp-python@llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/'"$CUDA_TAG"'@' requirements.txt
    fi
else
    echo "CUDA not detected"
fi

echo "________________________________________________"
echo "Stage 6: Setting up Python environment"
echo "________________________________________________"

pyenv local 3.12
echo "Creating virtual environment"
python -m venv venv
echo "Activating virtual environment"
source venv/bin/activate
echo "Upgrading pip, setuptools, and wheel"
python -m pip install --upgrade pip setuptools wheel
echo "Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo "________________________________________________"
echo "Installation complete"
echo "________________________________________________"
