@echo off
REM SkillBridge Environment Setup (CMD version)
REM Uses standard Windows Python to avoid MSYS2 build issues

echo ========================================
echo   SkillBridge Environment Setup
echo ========================================

py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python launcher (py) not found.
    echo Please install standard Python from https://www.python.org/downloads/
    exit /b 1
)

if exist .venv (
    echo Virtual environment already exists.
    set /p response="Remove and recreate? (y/n): "
    if /i "%response%"=="y" (
        rmdir /s /q .venv
        py -m venv .venv
    )
) else (
    echo Creating virtual environment...
    py -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate: .venv\Scripts\activate.bat
echo To run app:  streamlit run app.py
pause
