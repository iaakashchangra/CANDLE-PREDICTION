#!/usr/bin/env python
"""
TensorFlow Test Script for Apple Silicon (M1/M2/M3) Macs

This script verifies that TensorFlow is properly installed and can utilize the Metal GPU.
Run this script after setting up TensorFlow using the setup_tensorflow_mac.sh script.

Usage:
    conda activate tf_metal
    python test_tensorflow.py
"""

import os
import time
import numpy as np

# Set TF logging level to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Testing TensorFlow installation...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow successfully imported - version {tf.__version__}")
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU is available: {gpus}")
        
        # Enable memory growth to avoid allocating all GPU memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled for GPU")
        except RuntimeError as e:
            print(f"⚠️ Error setting memory growth: {e}")
    else:
        print("❌ No GPU found. TensorFlow will run on CPU only.")
        print("   If you're on Apple Silicon, make sure you've installed tensorflow-metal.")
    
    # Test with a simple model
    print("\nRunning a simple benchmark test...")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Generate random data
    x_train = np.random.random((1000, 100))
    y_train = np.random.randint(10, size=(1000,))
    
    # Time the training
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print("\n✅ TensorFlow is working correctly with GPU acceleration!")
    
    # Additional information
    print("\nSystem Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Numpy version: {np.__version__}")
    
    # Check if Metal plugin is available
    metal_available = any('Metal' in gpu.name for gpu in tf.config.list_physical_devices('GPU'))
    if metal_available:
        print("Metal plugin is being used for GPU acceleration")
    
    print("\nYour TensorFlow installation is ready to use!")
    
except ImportError as e:
    print(f"❌ Error importing TensorFlow: {e}")
    print("\nPlease make sure you've run the setup_tensorflow_mac.sh script and activated the environment:")
    print("  conda activate tf_metal")
    print("  python test_tensorflow.py")