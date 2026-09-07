# SkillBridge Environment Setup Script
# Uses standard Windows Python (via py launcher) to avoid MSYS2 build issues

$ErrorActionPreference = "Stop"
$VenvDir = ".venv"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SkillBridge Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check py launcher is available
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "ERROR: Python launcher (py) not found." -ForegroundColor Red
    Write-Host "Please install standard Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found Python launcher: $($py.Source)" -ForegroundColor Green
py --version

# Create virtual environment
if (Test-Path $VenvDir) {
    Write-Host "Virtual environment already exists at .\$VenvDir" -ForegroundColor Yellow
    $response = Read-Host "Remove and recreate? (y/n)"
    if ($response -eq 'y') {
        Remove-Item -Recurse -Force $VenvDir
        Write-Host "Creating fresh virtual environment..." -ForegroundColor Cyan
        py -m venv $VenvDir
    }
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -m venv $VenvDir
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$activateScript = ".\$VenvDir\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "ERROR: Activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in future sessions, run:" -ForegroundColor Yellow
Write-Host "    .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run the app:" -ForegroundColor Yellow
Write-Host "    streamlit run app.py" -ForegroundColor White
Write-Host ""

