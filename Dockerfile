# Use Python 3.9 image
FROM python:3.9-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies, including Redis and sudo
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libmagic1 \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    unrtf \
    poppler-utils \
    tesseract-ocr \
    flac \
    ffmpeg \
    lame \
    libmad0 \
    libsox-fmt-mp3 \
    sox \
    libjpeg-dev\
    swig \
    curl \
    redis-server \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Install latest Rust and Cargo using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file
COPY .env .

# Copy the rest of the application
COPY . .

# Expose the port the app runs on and Redis default port
EXPOSE 8089 6379

# Command to run Redis in the background and then the application
CMD redis-server & python3 swiss_army_llama.py
