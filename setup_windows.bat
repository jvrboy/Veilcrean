@echo off
echo Setting up Veilcrean on Windows...
python -m venv .venv
call .venv\Scripts\activate
REM requirements.txt only has the Vercel API deps - the brain needs the full stack.
pip install -r requirements.brain.txt
echo Setup Complete. To start the bot run: python run.py
pause
