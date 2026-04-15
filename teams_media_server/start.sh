#!/usr/bin/env bash
# Start the Teams Media Server (port 8001)
set -e
cd "$(dirname "$0")/.."
./venv/bin/uvicorn teams_media_server.server:app --host 0.0.0.0 --port 8001 --reload
