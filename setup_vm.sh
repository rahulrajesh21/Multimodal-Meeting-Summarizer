#!/bin/bash

# Vela - VM Setup Script for Linux (Ubuntu/Debian)
# This script installs system dependencies, Python environment, and Frontend dependencies.

set -e

echo "--- Updating system packages ---"
sudo apt-get update
sudo apt-get upgrade -y

echo "--- Installing system dependencies ---"
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    curl \
    git \
    lsof \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

echo "--- Installing Node.js 20.x ---"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "--- Setting up Python Virtual Environment ---"
python3 -m venv venv
source venv/bin/activate

echo "--- Installing Python dependencies ---"
# Upgrade pip first
pip install --upgrade pip
# Install requirements
pip install -r requirements.txt

echo "--- Setting up Frontend ---"
cd frontend
npm install
cd ..

echo "--- Creating a startup script ---"
cat > start_all.sh <<EOF
#!/bin/bash
# Script to start backend, frontend, and teams media server

# Kill anything already on these ports
echo "Clearing ports 8000, 8001, and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Activate virtual environment
source venv/bin/activate

# Start Backend in background
echo "Starting Backend on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=\$!

# Start Teams Media Server in background
echo "Starting Teams Media Server on port 8001..."
uvicorn teams_media_server.server:app --host 0.0.0.0 --port 8001 &
TEAMS_SERVER_PID=\$!

# Start Frontend in background
echo "Starting Frontend on port 3000..."
cd frontend
npm run dev -- -p 3000 &
FRONTEND_PID=\$!

echo "Vela is running!"
echo "Backend PID: \$BACKEND_PID"
echo "Teams Server PID: \$TEAMS_SERVER_PID"
echo "Frontend PID: \$FRONTEND_PID"
echo "Press Ctrl+C to stop all services."

# Handle shutdown
trap "kill \$BACKEND_PID \$TEAMS_SERVER_PID \$FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
EOF

chmod +x start_all.sh

echo "-------------------------------------------------------"
echo "Setup Complete!"
echo "To start the application, run: ./start_all.sh"
echo "Note: Ensure you have an LLM server (like LM Studio) running if required by your configuration."
echo "-------------------------------------------------------"
