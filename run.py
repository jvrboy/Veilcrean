"""
Veilcrean — Cross-Platform Entry Point
=====================================
Supports Windows (CMD/PowerShell), Linux (Bash), and Google Colab.
"""
import os
import sys
import subprocess
import platform

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_colab() -> bool:
    """Detect a Google Colab runtime reliably (import-based, not sys.modules)."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def setup_environment():
    print(f"Detecting platform: {platform.system()}...")

    # Check if running in Google Colab
    if is_colab():
        print("Running in Google Colab. Installing Veilcrean brain dependencies...")
        # Colab already ships numpy/pandas/torch, but we install the full brain
        # stack so every script works out of the box. `requirements.txt` only
        # contains the Vercel API deps — the brain needs requirements.brain.txt.
        req_file = os.path.join(REPO_ROOT, "requirements.brain.txt")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
            check=False,
        )
        return

    # Check for Virtual Env
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: You are not running in a virtual environment.")
        print("It is highly recommended to run: python -m venv .venv && source .venv/bin/activate")


def start_bot():
    print("Starting Veilcrean Python Brain...")
    # Add project root to path
    sys.path.append(REPO_ROOT)

    from python_brain.main import main
    sys.exit(main())


if __name__ == "__main__":
    setup_environment()
    start_bot()
