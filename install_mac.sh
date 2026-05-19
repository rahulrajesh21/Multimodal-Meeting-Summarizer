#!/usr/bin/env bash
# ============================================================
#  Vela — Multimodal Meeting Summarizer
#  macOS Installation Script
#  Run this once to set up the full project.
#  Requires: macOS, internet access
# ============================================================

set -e

# ── Colour helpers ──────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "\n============================================================"
echo -e " Vela  |  Multimodal Meeting Summarizer  |  macOS Setup"
echo -e "============================================================\n"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Check for Homebrew ───────────────────────────────────
echo -e "${CYAN}[1/5] Checking for Homebrew Package Manager...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}     Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for the current script if necessary
    if [[ -d "/opt/homebrew/bin" ]]; then
        export PATH="/opt/homebrew/bin:$PATH"
    fi
else
    echo -e "${GREEN}     Homebrew found.${NC}"
fi

# ── 2. Install System Dependencies ──────────────────────────
echo -e "${CYAN}[2/5] Installing System Dependencies (Python, Node.js, FFmpeg)...${NC}"
echo -e "      Updating Homebrew..."
brew update >/dev/null

echo -e "      Installing Python 3.11, Node.js, and FFmpeg..."
brew install python@3.11 node ffmpeg || echo -e "${YELLOW}     Some dependencies might already be installed.${NC}"

# Ensure python3.11 is available
PYTHON_BIN="python3.11"
if ! command -v $PYTHON_BIN &> /dev/null; then
    PYTHON_BIN="python3"
fi

# ── 3. Create Virtual Environment ───────────────────────────
echo -e "${CYAN}[3/5] Creating Python virtual environment...${NC}"
cd "$ROOT"
if [ -d "venv" ]; then
    echo -e "${YELLOW}     venv already exists — skipping creation.${NC}"
else
    $PYTHON_BIN -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}     Failed to create venv. Check your Python installation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}     Virtual environment created at ./venv${NC}"
fi

# ── 4. Install Python Dependencies ──────────────────────────
echo -e "${CYAN}[4/5] Installing Python packages...${NC}"
source "$ROOT/venv/bin/activate"

echo -e "      Upgrading pip..."
pip install --upgrade pip --no-cache-dir >/dev/null

echo -e "      Installing packages from requirements.txt..."
pip install -r "$ROOT/requirements.txt" --no-cache-dir || echo -e "${YELLOW}     Some packages failed to install. Continuing anyway...${NC}"
echo -e "${GREEN}     Python dependencies installed.${NC}"

# ── 5. Install Node Dependencies ────────────────────────────
echo -e "${CYAN}[5/5] Installing frontend Node dependencies...${NC}"
cd "$ROOT/frontend"
npm install --legacy-peer-deps
if [ $? -ne 0 ]; then
    echo -e "${RED}     npm install failed. Check Node.js version.${NC}"
    exit 1
fi
echo -e "${GREEN}     Node modules installed.${NC}"
cd "$ROOT"

# ── 6. Environment Setup ────────────────────────────────────
echo -e "${CYAN}[6/6] Setting up environment files...${NC}"

# Root .env
if [ ! -f "$ROOT/.env" ]; then
    if [ -f "$ROOT/.env.example" ]; then
        cp "$ROOT/.env.example" "$ROOT/.env"
        echo -e "${YELLOW}     Created .env from .env.example — please fill in your API keys.${NC}"
    else
        echo -e "${YELLOW}     No .env.example found at root. Creating blank .env...${NC}"
        echo "# Add your environment variables here" > "$ROOT/.env"
    fi
else
    echo -e "      Root .env already exists — skipping."
fi

# Frontend .env.local
if [ ! -f "$ROOT/frontend/.env.local" ]; then
    if [ -f "$ROOT/frontend/.env.example" ]; then
        cp "$ROOT/frontend/.env.example" "$ROOT/frontend/.env.local"
        echo -e "${YELLOW}     Created frontend/.env.local from .env.example${NC}"
    else
        echo -e "${YELLOW}     Creating frontend/.env.local with defaults...${NC}"
        cat <<EOF > "$ROOT/frontend/.env.local"
NEXT_PUBLIC_API_URL=http://localhost:8000

# Clerk Authentication — get keys from https://dashboard.clerk.com
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=

# Clerk redirect URLs
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
EOF
    fi
else
    echo -e "      frontend/.env.local already exists — skipping."
fi

# ── DONE ────────────────────────────────────────────────────
echo -e "\n============================================================"
echo -e "${GREEN}   Installation complete!${NC}"
echo -e "============================================================\n"
echo -e "   Next steps:\n"
echo -e "   1. Fill in your API keys:"
echo -e "      • $ROOT/.env"
echo -e "      • $ROOT/frontend/.env.local"
echo -e "        (NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY and CLERK_SECRET_KEY"
echo -e "         from https://dashboard.clerk.com)\n"
echo -e "   2. Start the app:"
echo -e "      Run: ./start.sh  OR run manually:\n"
echo -e "        Terminal 1 (API):"
echo -e "          source venv/bin/activate"
echo -e "          uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n"
echo -e "        Terminal 2 (Frontend):"
echo -e "          cd frontend && npm run dev\n"
echo -e "   3. Open  http://localhost:3000  in your browser\n"
echo -e "============================================================\n"
