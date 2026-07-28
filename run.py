"""
Veilcrean — Cross-Platform Entry Point
=====================================
Supports Windows (CMD/PowerShell), Linux (Bash), and Google Colab.
"""
import os
import sys
import subprocess
import platform

def setup_environment():
    print(f"Detecting platform: {platform.system()}...")
    
    # Check if running in Google Colab
    in_colab = 'google.colab' in sys.modules
    if in_colab:
        print("Running in Google Colab. Installing dependencies...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        return

    # Check for Virtual Env
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: You are not running in a virtual environment.")
        print("It is highly recommended to run: python -m venv .venv && source .venv/bin/activate")

def start_bot():
    print("Starting Veilcrean Python Brain...")
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from python_brain.main import main
    sys.exit(main())

if __name__ == "__main__":
    setup_environment()
    start_bot()
