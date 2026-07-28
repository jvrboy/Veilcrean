#!/bin/bash
echo "Setting up Veilcrean DSI Master Environment..."

# Create Venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install pandas numpy torch websockets deriv-api scipy scikit-learn python-dotenv

# Instructions
echo "===================================================="
echo "SETUP COMPLETE."
echo "1. Put your DERIV_API_TOKEN in .env"
echo "2. Run: python MASTER_DSI_TRAINER.py"
echo "===================================================="
