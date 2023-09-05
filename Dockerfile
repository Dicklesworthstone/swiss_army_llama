# Use Ubuntu latest image
ARG BASE_IMAGE
ARG ARCH
FROM --platform=linux/$ARCH $BASE_IMAGE

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies, including sudo
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    libpq-dev \
    libmagic1 \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Install CMake from the Ubuntu repositories
RUN apt-get update && \
    apt-get install -y cmake

# Confirm the installed CMake version
RUN cmake --version

RUN apt-get update && \
    apt-get install -y python3 python3-pip python-is-python3

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

# Expose the port the app runs on
EXPOSE 8089

# Command to run the application
CMD ["python3", "llama_2_embeddings_fastapi_server.py"]
