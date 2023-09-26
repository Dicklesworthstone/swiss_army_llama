#!/bin/bash

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install Docker
echo "Installing Docker..."
sudo apt-get install docker.io -y

# Start Docker service
echo "Starting Docker service..."
sudo systemctl start docker

# Display Docker version
echo "Checking Docker version..."
sudo docker --version

# Add the current user to the docker group
echo "Adding current user to the Docker group..."
sudo usermod -aG docker $USER

# Remove the old directory if it exists
echo "Removing old llama_embeddings_fastapi_service directory..."
rm -rf llama_embeddings_fastapi_service

# Clone the repository
echo "Cloning the llama_embeddings_fastapi_service repository..."
git clone https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service

# Change to the repository directory
cd llama_embeddings_fastapi_service

# Build the Docker image
echo "Building the Docker image..."
arch=$(uname -m)
base_image="ubuntu:latest"

if [ "$arch" = "x86_64" ]; then
  echo "Building for x86_64..."
  sudo docker build --build-arg BASE_IMAGE=$base_image --build-arg ARCH="amd64" -t llama-embeddings .
elif [ "$arch" = "aarch64" ]; then
  echo "Building for aarch64..."
  sudo docker build --build-arg BASE_IMAGE=$base_image --build-arg ARCH="arm64" -t llama-embeddings .
else
  echo "Unsupported architecture."
  exit 1
fi


# Run the Docker container
echo "Running the Docker container..."
sudo docker run -e TERM=$TERM -p 8089:8089 llama-embeddings

echo "Script completed!"
