@echo off
echo Setting up Veilcrean on Windows...
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
echo Setup Complete. To start the bot run: python run.py
pause
