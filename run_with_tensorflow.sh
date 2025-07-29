#!/bin/bash

# Script to run the application with TensorFlow support
# This script activates the tf_metal conda environment and runs the application

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if the tf_metal environment exists
if ! conda env list | grep -q "tf_metal"; then
    echo "Error: tf_metal environment not found."
    echo "Please run the setup_tensorflow_mac.sh script first to create the environment."
    exit 1
fi

# Activate the conda environment
echo "Activating tf_metal environment..."
# Use source activate which works better in scripts
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tf_metal

# Check if required packages are installed
echo "Checking required packages..."
if ! pip list | grep -q "flask"; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Run the application
echo "Running application with TensorFlow support..."
python app.py