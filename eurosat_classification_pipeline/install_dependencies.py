#!/usr/bin/env python3
# ============================================================================
# install_dependencies.py
# Description: Automatic installation script for all required libraries
# Usage: python install_dependencies.py
# ============================================================================

import subprocess
import sys
import os

def install_package(package):
    """Install a single package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed: {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\n" + "="*60)
    print("📦 Upgrading pip to latest version...")
    print("="*60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully!\n")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to upgrade pip, continuing anyway...\n")

def main():
    """Main installation function"""
    
    print("\n" + "="*60)
    print("🚀 EuroSAT Project - Dependency Installation Script")
    print("="*60)
    
    # List of required packages
    packages = [
        # Core Scientific Computing
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        
        # Image Processing
        "scikit-image>=0.18.0",
        "tifffile>=2021.0.0",
        "Pillow>=8.0.0",
        
        # Machine Learning
        "scikit-learn>=0.24.0",
        
        # Deep Learning
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        
        # Data Download
        "kagglehub>=0.1.0",
        
        # Visualization
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        
        # Progress bars
        "tqdm>=4.60.0",
        
        # Utilities
        "python-dateutil>=2.8.0",
        "pytz>=2021.1",
    ]
    
    # Upgrade pip first
    upgrade_pip()
    
    # Install packages
    print("="*60)
    print("📥 Installing required packages...")
    print("="*60 + "\n")
    
    successful = []
    failed = []
    
    total_packages = len(packages)
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{total_packages}] Installing {package}...")
        print("-" * 60)
        
        if install_package(package):
            successful.append(package)
        else:
            failed.append(package)
    
    # Summary
    print("\n" + "="*60)
    print("📊 INSTALLATION SUMMARY")
    print("="*60)
    print(f"✅ Successfully installed: {len(successful)}/{total_packages}")
    print(f"❌ Failed to install: {len(failed)}/{total_packages}")
    
    if successful:
        print("\n✅ Successfully installed packages:")
        for pkg in successful:
            print(f"   • {pkg}")
    
    if failed:
        print("\n❌ Failed packages (try installing manually):")
        for pkg in failed:
            print(f"   • {pkg}")
        print("\n💡 To install failed packages manually, run:")
        print(f"   pip install {' '.join(failed)}")
    
    print("\n" + "="*60)
    
    if not failed:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("✅ You can now run the main project!")
        print("\n📝 Next steps:")
        print("   python main.py")
    else:
        print("⚠️ SOME PACKAGES FAILED TO INSTALL")
        print("Please install them manually before running the project.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        sys.exit(1)