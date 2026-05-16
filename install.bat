@echo off
setlocal EnableDelayedExpansion
title Vela — Windows Installer

:: ============================================================
::  Vela — Multimodal Meeting Summarizer
::  Windows Installation Script
::  Run this once to set up the full project.
::  Requires: Windows 10/11, internet access
:: ============================================================

echo.
echo  ============================================================
echo   Vela  ^|  Multimodal Meeting Summarizer  ^|  Windows Setup
echo  ============================================================
echo.

:: ── Root of the repo ─────────────────────────────────────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: ── Colour helpers (using ANSI — works on Win10+) ────────────
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "CYAN=[36m"
set "RESET=[0m"

goto :CHECK_ADMIN

:: ────────────────────────────────────────────────────────────
:CHECK_ADMIN
net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %RED%[!] This script requires Administrator privileges.%RESET%
    echo     Right-click install.bat ^> "Run as administrator"
    pause
    exit /b 1
)

:: ────────────────────────────────────────────────────────────
:CHECK_PYTHON
echo %CYAN%[1/7] Checking Python 3.10+...%RESET%
python --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %YELLOW%     Python not found. Opening download page...%RESET%
    echo     Please install Python 3.10 or newer from https://www.python.org/downloads/
    echo     IMPORTANT: Check "Add Python to PATH" during installation.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo %GREEN%     Found Python %PY_VER%%RESET%

:: ────────────────────────────────────────────────────────────
:CHECK_NODE
echo %CYAN%[2/7] Checking Node.js 20+...%RESET%
node --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %YELLOW%     Node.js not found. Opening download page...%RESET%
    echo     Please install Node.js 20 LTS from https://nodejs.org/
    start https://nodejs.org/en/download
    pause
    exit /b 1
)
for /f %%v in ('node --version') do set "NODE_VER=%%v"
echo %GREEN%     Found Node.js %NODE_VER%%RESET%

:: ────────────────────────────────────────────────────────────
:CHECK_FFMPEG
echo %CYAN%[3/7] Checking FFmpeg...%RESET%
ffmpeg -version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %YELLOW%     FFmpeg not found.%RESET%
    echo.
    echo     Attempting to install FFmpeg via winget...
    winget install --id Gyan.FFmpeg -e --silent >nul 2>&1
    if !errorlevel! NEQ 0 (
        echo %RED%     winget install failed. Please install FFmpeg manually:%RESET%
        echo     1. Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-essentials.zip)
        echo     2. Extract to C:\ffmpeg
        echo     3. Add C:\ffmpeg\bin to your PATH environment variable
        echo     4. Re-run this script
        start https://www.gyan.dev/ffmpeg/builds/
        pause
        exit /b 1
    )
    :: Refresh PATH after winget install
    call RefreshEnv.cmd >nul 2>&1
    ffmpeg -version >nul 2>&1
    if !errorlevel! NEQ 0 (
        echo %YELLOW%     FFmpeg installed but not yet on PATH.%RESET%
        echo     Please restart your terminal and re-run this script, OR
        echo     add FFmpeg to PATH manually, then continue.
        pause
    )
    echo %GREEN%     FFmpeg installed successfully.%RESET%
) else (
    echo %GREEN%     FFmpeg found.%RESET%
)

:: ────────────────────────────────────────────────────────────
:VENV
echo %CYAN%[4/7] Creating Python virtual environment...%RESET%
if exist "%ROOT%\venv" (
    echo %YELLOW%     venv already exists — skipping creation.%RESET%
) else (
    python -m venv "%ROOT%\venv"
    if !errorlevel! NEQ 0 (
        echo %RED%     Failed to create venv. Check your Python installation.%RESET%
        pause & exit /b 1
    )
    echo %GREEN%     Virtual environment created at .\venv%RESET%
)

:: ────────────────────────────────────────────────────────────
:PIP_INSTALL
echo %CYAN%[5/7] Installing Python dependencies (this may take a few minutes)...%RESET%
echo     Upgrading pip...
"%ROOT%\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo     Installing packages from requirements.txt...
"%ROOT%\venv\Scripts\pip.exe" install -r "%ROOT%\requirements.txt" --quiet
if %errorlevel% NEQ 0 (
    echo %RED%     Some packages failed to install.%RESET%
    echo     Common fixes:
    echo       • PyAudio: install PortAudio via  pip install pipwin ^&^& pipwin install pyaudio
    echo       • torch: visit https://pytorch.org/get-started/locally/ for GPU builds
    echo     You can re-run this step with: venv\Scripts\pip install -r requirements.txt
    echo.
    echo     Continuing with remaining setup...
)

:: PyAudio Windows fallback via pipwin
"%ROOT%\venv\Scripts\python.exe" -c "import pyaudio" >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %YELLOW%     PyAudio not importable — trying pipwin fallback...%RESET%
    "%ROOT%\venv\Scripts\pip.exe" install pipwin --quiet
    "%ROOT%\venv\Scripts\pipwin.exe" install pyaudio --quiet
)

echo %GREEN%     Python dependencies installed.%RESET%

:: ────────────────────────────────────────────────────────────
:NPM_INSTALL
echo %CYAN%[6/7] Installing frontend Node dependencies...%RESET%
cd /d "%ROOT%\frontend"
call npm install --legacy-peer-deps
if %errorlevel% NEQ 0 (
    echo %RED%     npm install failed. Check Node.js version (requires 20+).%RESET%
    cd /d "%ROOT%"
    pause & exit /b 1
)
echo %GREEN%     Node modules installed.%RESET%
cd /d "%ROOT%"

:: ────────────────────────────────────────────────────────────
:ENV_SETUP
echo %CYAN%[7/7] Setting up environment files...%RESET%

:: Root .env
if not exist "%ROOT%\.env" (
    if exist "%ROOT%\.env.example" (
        copy "%ROOT%\.env.example" "%ROOT%\.env" >nul
        echo %YELLOW%     Created .env from .env.example — please fill in your API keys.%RESET%
    ) else (
        echo %YELLOW%     No .env.example found at root. Creating blank .env...%RESET%
        echo # Add your environment variables here > "%ROOT%\.env"
    )
) else (
    echo      Root .env already exists — skipping.
)

:: Frontend .env.local
if not exist "%ROOT%\frontend\.env.local" (
    if exist "%ROOT%\frontend\.env.example" (
        copy "%ROOT%\frontend\.env.example" "%ROOT%\frontend\.env.local" >nul
        echo %YELLOW%     Created frontend\.env.local from .env.example%RESET%
    ) else (
        echo %YELLOW%     Creating frontend\.env.local with defaults...%RESET%
        (
            echo NEXT_PUBLIC_API_URL=http://localhost:8000
            echo.
            echo # Clerk Authentication — get keys from https://dashboard.clerk.com
            echo NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
            echo CLERK_SECRET_KEY=
            echo.
            echo # Clerk redirect URLs
            echo NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
            echo NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
            echo NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
            echo NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
        ) > "%ROOT%\frontend\.env.local"
    )
) else (
    echo      frontend\.env.local already exists — skipping.
)

:: ────────────────────────────────────────────────────────────
:DONE
echo.
echo  ============================================================
echo %GREEN%   Installation complete!%RESET%
echo  ============================================================
echo.
echo   Next steps:
echo.
echo   1. Fill in your API keys:
echo      • %ROOT%\.env
echo      • %ROOT%\frontend\.env.local
echo        (NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY and CLERK_SECRET_KEY
echo         from https://dashboard.clerk.com)
echo.
echo   2. Start the app:
echo      Double-click  start.bat  OR run manually:
echo.
echo        Terminal 1 (API):
echo          venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo        Terminal 2 (Frontend):
echo          cd frontend ^&^& npm run dev
echo.
echo   3. Open  http://localhost:3000  in your browser
echo.
echo  ============================================================
echo.
pause
endlocal
