#!/usr/bin/env python3
"""
Setup check script for the Stress Granule Analysis project.
This script verifies that all required dependencies are installed and provides
helpful information about the project structure.
"""

import sys
import os
import importlib
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    major, minor = sys.version_info[:2]
    print(f"Python version: {major}.{minor}")
    
    if major < 3 or (major == 3 and minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: not installed")
        return False

def check_dependencies():
    """Check all required dependencies"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('opencv-python', 'cv2'),
        ('tqdm', 'tqdm'),
        ('pandas', 'pandas')
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    return all_installed

def check_gpu_availability():
    """Check if GPU is available for PyTorch"""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            
            print(f"✅ GPU available: {gpu_name}")
            print(f"   GPU count: {gpu_count}")
            print(f"   Memory: {memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU available, will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not available, cannot check GPU")
        return False

def check_project_structure():
    """Check if project files are present"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'main.py',
        'models.py',
        'metrics.py',
        'inference.py',
        'requirements.txt',
        'README.md',
        'data/images/.gitkeep',
        'data/masks/.gitkeep'
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: missing")
            all_present = False
    
    return all_present

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📖 Usage Instructions:")
    print("="*50)
    
    print("\n1. Training a model:")
    print("   python main.py --data_dir path/to/data --epochs 100")
    
    print("\n2. Running inference:")
    print("   python inference.py --model best_stress_granule_model.pth --input_dir path/to/images")
    
    print("\n3. Data organization:")
    print("   Place your images in: data/images/")
    print("   Place your masks in:  data/masks/")
    
    print("\n4. View training metrics:")
    print("   Check the metrics/ directory after training")
    
    print("\n5. Command line help:")
    print("   python main.py --help")
    print("   python inference.py --help")

def main():
    """Main setup check function"""
    print("🧬 Stress Granule Analysis - Setup Check")
    print("="*50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check GPU
    gpu_ok = check_gpu_availability()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Print summary
    print("\n📊 Summary:")
    print("="*20)
    
    if python_ok and deps_ok and structure_ok:
        print("✅ Setup is complete! You can start training.")
        if gpu_ok:
            print("🚀 GPU detected - training will be fast!")
        else:
            print("⚠️  No GPU detected - training will be slower but functional.")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        
        if not deps_ok:
            print("\n💡 To install missing dependencies:")
            print("   pip install -r requirements.txt")
        
        if not structure_ok:
            print("\n💡 Make sure you're in the project root directory")
    
    # Print usage instructions regardless
    print_usage_instructions()

if __name__ == "__main__":
    main() 