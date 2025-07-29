#!/usr/bin/env python
"""
Dependency Checker for TensorFlow Environment

This script checks if all the required packages are installed in the current environment.
Run this script after setting up the tf_metal environment to verify all dependencies.

Usage:
    conda activate tf_metal
    python check_dependencies.py
"""

import importlib.util
import sys
import os

# List of required packages
REQUIRED_PACKAGES = [
    # Core web framework
    'flask',
    'werkzeug',
    'jinja2',
    
    # Data manipulation
    'pandas',
    'numpy',
    
    # Machine learning
    'tensorflow',  # Should be installed as tensorflow-macos on Apple Silicon
    'scikit-learn',
    'keras',
    
    # Technical analysis
    'ta',
    
    # Data visualization
    'matplotlib',
    'seaborn',
    
    # API requests
    'requests',
    
    # Database
    'sqlalchemy',
    
    # Data preprocessing
    'scipy',
    
    # Configuration
    'pyyaml',
    
    # Async support
    'aiohttp',
    
    # Date/time handling
    'python-dateutil',
]

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    return True

def get_package_version(package_name):
    """Get the version of an installed package"""
    try:
        if package_name == 'tensorflow':
            # Special handling for tensorflow which might be installed as tensorflow-macos
            import tensorflow
            return tensorflow.__version__
        else:
            module = importlib.import_module(package_name)
            return getattr(module, '__version__', 'Unknown')
    except (ImportError, AttributeError):
        return 'Not installed'

def main():
    print("\n=== Dependency Checker for TensorFlow Environment ===\n")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check if running in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Conda environment: {conda_env}")
    else:
        print("Warning: Not running in a conda environment")
    
    # Check TensorFlow and GPU
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {gpus}")
            # Check if Metal plugin is available
            metal_available = any('Metal' in gpu.name for gpu in gpus) if hasattr(gpus[0], 'name') else False
            if metal_available:
                print("✅ Metal plugin is being used for GPU acceleration")
            else:
                print("⚠️ GPU is available but Metal plugin is not detected")
        else:
            print("❌ No GPU found. TensorFlow will run on CPU only.")
    except ImportError:
        print("❌ TensorFlow not installed")
    
    print("\nChecking required packages:\n")
    
    missing_packages = []
    all_packages_info = []
    
    # Check each required package
    for package in REQUIRED_PACKAGES:
        is_installed = check_package(package)
        version = get_package_version(package) if is_installed else 'Not installed'
        
        if is_installed:
            status = "✅"
        else:
            status = "❌"
            missing_packages.append(package)
        
        all_packages_info.append((package, version, status))
    
    # Print package information in a table format
    print(f"{'Package':<20} {'Version':<15} {'Status':<10}")
    print("-" * 45)
    
    for package, version, status in all_packages_info:
        print(f"{package:<20} {version:<15} {status:<10}")
    
    # Summary
    print("\nSummary:")
    print(f"Total packages required: {len(REQUIRED_PACKAGES)}")
    print(f"Packages installed: {len(REQUIRED_PACKAGES) - len(missing_packages)}")
    print(f"Packages missing: {len(missing_packages)}")
    
    if missing_packages:
        print("\nMissing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nTo install missing packages, run:")
        print("pip install " + " ".join(missing_packages))
    else:
        print("\n✅ All required packages are installed!")

if __name__ == "__main__":
    main()