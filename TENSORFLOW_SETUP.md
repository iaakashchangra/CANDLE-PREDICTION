# TensorFlow Setup for Apple Silicon (M1/M2/M3) Macs

This guide provides instructions for setting up TensorFlow with GPU acceleration on Apple Silicon Macs (M1, M2, M3 chips).

## Prerequisites

- macOS 12 (Monterey) or later
- Conda package manager (Miniconda or Anaconda)

## Automatic Setup

We've provided a script that automates the setup process:

```bash
# Make the script executable (if not already)
chmod +x setup_tensorflow_mac.sh

# Run the setup script
./setup_tensorflow_mac.sh
```

The script will:
1. Create a new conda environment called `tf_metal` with Python 3.10
2. Install TensorFlow dependencies
3. Install TensorFlow for macOS and the Metal plugin
4. Verify the installation

## Manual Setup

If you prefer to set up TensorFlow manually, follow these steps:

1. Create a new conda environment with Python 3.10:
   ```bash
   conda create -n tf_metal python=3.10
   conda activate tf_metal
   ```

2. Install TensorFlow dependencies:
   ```bash
   conda install -c apple tensorflow-deps
   ```

3. Install TensorFlow for macOS and Metal plugin:
   ```bash
   pip install tensorflow-macos tensorflow-metal
   ```

4. Verify the installation:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
   ```

## Testing the Installation

We've included a test script to verify that TensorFlow is working correctly with GPU acceleration:

```bash
conda activate tf_metal
python test_tensorflow.py
```

The script will run a simple benchmark to confirm that TensorFlow is using the Metal GPU.

## Using TensorFlow with Your Project

To use TensorFlow in your project:

1. Activate the conda environment:
   ```bash
   conda activate tf_metal
   ```

2. Run your Python script:
   ```bash
   python app.py
   ```

3. Alternatively, you can create a shell script to automate this process:
   ```bash
   #!/bin/bash
   eval "$(conda shell.bash hook)"
   conda activate tf_metal
   python app.py
   ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named tensorflow**
   - Make sure you've activated the correct conda environment: `conda activate tf_metal`

2. **No GPU devices found**
   - Verify that tensorflow-metal is installed: `pip list | grep tensorflow-metal`
   - Make sure you're using Python 3.10 or 3.11: `python --version`

3. **Memory errors**
   - Add the following code to your script to limit GPU memory usage:
     ```python
     import tensorflow as tf
     gpus = tf.config.list_physical_devices('GPU')
     if gpus:
         for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
     ```

## Performance Tips

1. **Batch Size**: Adjust the batch size to optimize performance. Start with smaller batch sizes (16-32) and increase gradually.

2. **Mixed Precision**: Use mixed precision training to improve performance:
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

3. **Model Optimization**: Consider using TensorFlow Lite or model optimization techniques for inference.

## Additional Resources

- [TensorFlow Metal Plugin Documentation](https://developer.apple.com/metal/tensorflow-plugin/)
- [Apple Machine Learning Documentation](https://developer.apple.com/machine-learning/)
- [TensorFlow Documentation](https://www.tensorflow.org/)