#!/bin/bash

# Setup TensorFlow for Apple Silicon (M1/M2/M3) Macs
# This script creates a conda environment with TensorFlow and Metal support

echo "Setting up TensorFlow for Apple Silicon Mac..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create a new conda environment with Python 3.10
echo "Creating conda environment 'tf_metal' with Python 3.10..."
conda create -y -n tf_metal python=3.10

# Activate the environment
echo "Activating tf_metal environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tf_metal

# Install TensorFlow dependencies
echo "Installing TensorFlow dependencies..."
conda install -y -c apple tensorflow-deps

# Install TensorFlow for macOS and Metal plugin
echo "Installing TensorFlow for macOS and Metal plugin..."
pip install tensorflow-macos tensorflow-metal

# Verify installation
echo "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Install project dependencies
echo ""
echo "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    # Skip TensorFlow in requirements.txt since we've already installed it
    grep -v "tensorflow" requirements.txt > tf_requirements.txt
    pip install -r tf_requirements.txt
    rm tf_requirements.txt
    echo "Project dependencies installed successfully."
else
    echo "Warning: requirements.txt not found. Skipping project dependencies installation."
fi

echo ""
echo "TensorFlow setup complete!"
echo "To activate this environment, run: conda activate tf_metal"
echo "To use this environment with your project, add the following to your Python script:"
echo ""
echo "import tensorflow as tf"
echo "print('TensorFlow version:', tf.__version__)"
echo "print('GPU available:', tf.config.list_physical_devices('GPU'))"
echo ""
echo "To update your app.py to use this environment, modify the shebang line to:"
echo "#!/usr/bin/env python"
echo "And run your script with:"
echo "conda activate tf_metal && python app.py"