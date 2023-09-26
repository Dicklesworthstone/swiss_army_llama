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
echo "Removing old swiss_army_llama directory..."
rm -rf swiss_army_llama

# Clone the repository
echo "Cloning the swiss_army_llama repository..."
git clone https://github.com/Dicklesworthstone/swiss_army_llama

# Change to the repository directory
cd swiss_army_llama

# Build the Docker image
echo "Building the Docker image..."
arch=$(uname -m)
base_image="ubuntu:latest"

if [ "$arch" = "x86_64" ]; then
  echo "Building for x86_64..."
  sudo docker build --build-arg BASE_IMAGE=$base_image --build-arg ARCH="amd64" -t swiss-army-llama .
elif [ "$arch" = "aarch64" ]; then
  echo "Building for aarch64..."
  sudo docker build --build-arg BASE_IMAGE=$base_image --build-arg ARCH="arm64" -t  swiss-army-llama .
else
  echo "Unsupported architecture."
  exit 1
fi


# Run the Docker container
echo "Running the Docker container..."
sudo docker run -e TERM=$TERM -p 8089:8089 swiss-army-llama

echo "Script completed!"
