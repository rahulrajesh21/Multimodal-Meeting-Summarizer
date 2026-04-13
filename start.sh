#!/bin/bash
# MeetingIQ — start both servers
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 Starting MeetingIQ..."

# Kill anything already on these ports
echo "  → Clearing ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1


# 1. FastAPI backend
echo "  → Starting API on :8000"
"$ROOT/venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# 2. Next.js frontend (needs Node 20)
echo "  → Starting frontend on :3000"
source "$HOME/.nvm/nvm.sh" 2>/dev/null
nvm use 20 --silent 2>/dev/null || true
cd "$ROOT/frontend" && npm run dev -- --port 3000 &
FE_PID=$!

echo ""
echo "  ✅ MeetingIQ running!"
echo "     Frontend → http://localhost:3000"
echo "     API      → http://localhost:8000"
echo ""
echo "  Press Ctrl+C to stop both servers."

trap "kill $API_PID $FE_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
