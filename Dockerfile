FROM python:3.11-slim

WORKDIR /home/user/Veilcrean

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy full trading-brain requirements and install
COPY requirements.brain.txt .
RUN pip install --no-cache-dir -r requirements.brain.txt

# Copy source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/home/user/Veilcrean

# Start bot
CMD ["python", "run.py"]
