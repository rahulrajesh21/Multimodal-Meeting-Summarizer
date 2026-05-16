@echo off
setlocal EnableDelayedExpansion
title Vela — Starting Servers

:: ============================================================
::  Vela — Start Script (Windows)
::  Launches FastAPI backend + Next.js frontend in parallel.
::  Run install.bat first if you haven't already.
:: ============================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "CYAN=[36m"
set "RESET=[0m"

echo.
echo  ============================================================
echo   Vela  ^|  Starting Servers
echo  ============================================================
echo.

:: ── Sanity checks ────────────────────────────────────────────
if not exist "%ROOT%\venv\Scripts\uvicorn.exe" (
    echo %RED%[!] venv not found or uvicorn missing.%RESET%
    echo     Run install.bat first.
    pause & exit /b 1
)

if not exist "%ROOT%\frontend\node_modules" (
    echo %RED%[!] frontend\node_modules not found.%RESET%
    echo     Run install.bat first.
    pause & exit /b 1
)

if not exist "%ROOT%\frontend\.env.local" (
    echo %YELLOW%[!] frontend\.env.local not found.%RESET%
    echo     Clerk keys may be missing — the app may not authenticate correctly.
    echo     Run install.bat to generate it, then add your Clerk API keys.
    echo.
)

:: ── Kill anything already on 8000 / 3000 ─────────────────────
echo %CYAN%Freeing ports 8000 and 3000...%RESET%
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8001 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)

:: ── Launch FastAPI backend in a new window ────────────────────
echo %CYAN%Starting API server on http://localhost:8000 ...%RESET%
start "Vela — API (port 8000)" cmd /k "cd /d "%ROOT%" && venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: ── Launch Teams Media Server in a new window ───────────────────
echo %CYAN%Starting Teams Media Server on http://localhost:8001 ...%RESET%
start "Vela — Teams Media Server (port 8001)" cmd /k "cd /d "%ROOT%" && venv\Scripts\uvicorn teams_media_server.server:app --host 0.0.0.0 --port 8001 --reload"

:: Give the APIs a moment to bind
timeout /t 2 /nobreak >nul

:: ── Launch Next.js frontend in a new window ───────────────────
echo %CYAN%Starting frontend on http://localhost:3000 ...%RESET%
start "Vela — Frontend (port 3000)" cmd /k "cd /d "%ROOT%\frontend" && npm run dev -- --port 3000"

:: ── Wait then open browser ────────────────────────────────────
echo.
echo %YELLOW%Waiting 5 seconds for servers to initialise...%RESET%
timeout /t 5 /nobreak >nul

echo %GREEN%Opening http://localhost:3000 in your browser...%RESET%
start "" "http://localhost:3000"

echo.
echo  ============================================================
echo %GREEN%   Vela is running!%RESET%
echo  ============================================================
echo.
echo   Frontend  →  http://localhost:3000
echo   API       →  http://localhost:8000
echo   API docs  →  http://localhost:8000/docs
echo   Media Srv →  http://localhost:8001
echo.
echo   Three terminal windows are open — close them to stop the servers.
echo   (Or just close this window; those processes will keep running.)
echo.
pause
endlocal
