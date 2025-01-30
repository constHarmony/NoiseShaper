#!/bin/python3
import sys
import subprocess

def check_python_version():
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print("Python version: " + sys.version)

def upgrade_pip():
    try:
        print("Upgrading pip...")
        subprocess.check_call(["curl", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"])
        subprocess.check_call([sys.executable, "get-pip.py"])
        print("Pip upgraded successfully.")
    except subprocess.CalledProcessError:
        print("Failed to upgrade pip. Ensure your Python installation is correct.")
        sys.exit(1)

def install_dependencies():
    dependencies = [
        "numpy",
        "pyqt6",
        "pyqtgraph",
        "sounddevice",
        "soundfile",
        "scipy",
    ]
    try:
        print("Installing dependencies...")
        for dependency in dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install dependencies. Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    print("Starting setup...")
    check_python_version()
    upgrade_pip()
    install_dependencies()
    print("Setup complete. Run the app using: python3 app.py")
