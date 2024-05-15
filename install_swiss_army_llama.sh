#!/bin/bash

set -e

# Check for pyenv and install if not present
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
fi

# Detect default shell and add pyenv to the shell configuration file if not already present
DEFAULT_SHELL=$(basename "$SHELL")
if [ "$DEFAULT_SHELL" = "zsh" ]; then
    CONFIG_FILE="$HOME/.zshrc"
else
    CONFIG_FILE="$HOME/.bashrc"
fi

if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' "$CONFIG_FILE"; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$CONFIG_FILE"
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$CONFIG_FILE"
    echo 'eval "$(pyenv init --path)"' >> "$CONFIG_FILE"
    source "$CONFIG_FILE"
fi

cd ~/.pyenv && git pull && cd -
pyenv install 3.12

# Install additional dependencies
sudo apt-get update
sudo apt-get install -y libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig redis-server
sudo systemctl enable redis-server
sudo systemctl start redis

# Detect CUDA and update requirements.txt
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    case $CUDA_VERSION in
        12.1*) CUDA_TAG="cu121" ;;
        12.2*) CUDA_TAG="cu122" ;;
        12.3*) CUDA_TAG="cu123" ;;
        12.4*) CUDA_TAG="cu124" ;;
        *) CUDA_TAG="" ;;
    esac

    if [ -n "$CUDA_TAG" ]; then
        sed -i 's/faiss-cpu/faiss/' requirements.txt
        sed -i 's@llama-cpp-python@llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/'"$CUDA_TAG"'@' requirements.txt
    fi
fi

# Set up Python environment
pyenv local 3.12
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
