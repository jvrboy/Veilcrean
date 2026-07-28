#!/bin/bash
echo "Setting up Veilcrean on Linux..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup Complete. To start the bot run: python run.py"
