#!/usr/bin/env python3
"""
Simple script to verify that the KnowWhat environment is set up correctly.
Run this after following the setup instructions in README.md.
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python version: {sys.version.split()[0]}")
        return True
    else:
        print(f"✗ Python version: {sys.version.split()[0]} (need 3.10+)")
        return False

def check_virtual_env():
    """Check if running in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Running in virtual environment")
        return True
    else:
        print("⚠ Not running in virtual environment (recommended)")
        return False

def check_dependencies():
    """Check if core dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
        'statsmodels', 'sklearn', 'PIL', 'requests', 'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                import PIL
            elif package == 'dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_data_directories():
    """Check if required data directories exist"""
    required_dirs = [
        'data/experiment_mazes',
        'data/human_results',
        'data/machine_results'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    print("KnowWhat Environment Setup Check")
    print("=" * 35)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Virtual Environment", check_virtual_env()),
        ("Dependencies", check_dependencies()[0]),
        ("Data Directories", check_data_directories())
    ]
    
    print("\nSummary:")
    print("-" * 20)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 Environment setup complete! You can run 'make analysis'")
    else:
        print("⚠  Some issues found. Please check the setup instructions in README.md")
        
    # Check for missing dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\nTo install missing dependencies: pip install {' '.join(missing)}")

if __name__ == "__main__":
    main()
