#!/bin/bash
echo "Setting up Veilcrean DSI Master Environment..."

# Create Venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements (deriv-api is not on PyPI; the repo talks to Deriv
# directly over WebSockets via the `websockets` package)
pip install --upgrade pip
pip install -r requirements.brain.txt

# Instructions
echo "===================================================="
echo "SETUP COMPLETE."
echo "1. Put your DERIV_API_TOKEN in .env"
echo "2. Run: python MASTER_DSI_TRAINER.py"
echo "===================================================="
