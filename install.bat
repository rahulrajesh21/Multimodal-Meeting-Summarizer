@echo off
setlocal EnableDelayedExpansion
title Vela — Windows Installer (Chocolatey)

:: ============================================================
::  Vela — Multimodal Meeting Summarizer
::  Windows Installation Script (Fully Automated via Chocolatey)
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
    echo %RED%[!] This script requires Administrator privileges to install Chocolatey and system packages.%RESET%
    echo     Right-click install.bat ^> "Run as administrator"
    pause
    exit /b 1
)

:: ────────────────────────────────────────────────────────────
:CHOCO_SETUP
echo %CYAN%[1/5] Checking for Chocolatey Package Manager...%RESET%
choco -v >nul 2>&1
if %errorlevel% NEQ 0 (
    echo %YELLOW%     Chocolatey not found. Installing Chocolatey...%RESET%
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if !errorlevel! NEQ 0 (
        echo %RED%     Failed to install Chocolatey. Please install manually from https://chocolatey.org/install%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%     Chocolatey installed successfully.%RESET%
    :: Refresh path in current session so choco is available immediately
    call "%ALLUSERSPROFILE%\chocolatey\bin\RefreshEnv.cmd" >nul 2>&1
) else (
    echo %GREEN%     Chocolatey found.%RESET%
)

:: ────────────────────────────────────────────────────────────
:INSTALL_DEPENDENCIES
echo %CYAN%[2/5] Installing System Dependencies (Python, Node.js, FFmpeg) silently...%RESET%
echo     This may take a few minutes. Please wait...
choco install python --version=3.11.9 --allow-downgrade -y
choco install nodejs ffmpeg -y
if %errorlevel% NEQ 0 (
    echo %RED%     Failed to install some system dependencies via Chocolatey.%RESET%
    pause
    exit /b 1
)
echo %GREEN%     System dependencies installed.%RESET%

:: Refresh path so Python, Node, and FFmpeg are available immediately
call "%ALLUSERSPROFILE%\chocolatey\bin\RefreshEnv.cmd" >nul 2>&1

:: ────────────────────────────────────────────────────────────
:VENV
echo %CYAN%[3/5] Creating Python virtual environment...%RESET%
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
echo %CYAN%[4/5] Installing Python packages...%RESET%
echo     Upgrading pip...
"%ROOT%\venv\Scripts\python.exe" -m pip install --upgrade pip --no-cache-dir

echo     Installing packages from requirements.txt (this will take several minutes to download large ML models)...
"%ROOT%\venv\Scripts\pip.exe" install -r "%ROOT%\requirements.txt" --no-cache-dir
if %errorlevel% NEQ 0 (
    echo %YELLOW%     Some packages failed to install. Continuing anyway...%RESET%
)



echo %GREEN%     Python dependencies installed.%RESET%

:: ────────────────────────────────────────────────────────────
:NPM_INSTALL
echo %CYAN%[5/5] Installing frontend Node dependencies...%RESET%
cd /d "%ROOT%\frontend"
call npm install --legacy-peer-deps
if %errorlevel% NEQ 0 (
    echo %RED%     npm install failed. Check Node.js version.%RESET%
    cd /d "%ROOT%"
    pause & exit /b 1
)
echo %GREEN%     Node modules installed.%RESET%
cd /d "%ROOT%"

:: ────────────────────────────────────────────────────────────
:ENV_SETUP
echo %CYAN%[6/6] Setting up environment files...%RESET%

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
