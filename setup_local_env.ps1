# Setup script for Windows to create traj_opt_env virtual environment
# Run this script with: .\setup_local_env.ps1

Write-Host "Setting up trajectory optimization environment..." -ForegroundColor Green

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Display Python version
$pythonVersion = python --version
Write-Host "Using $pythonVersion" -ForegroundColor Cyan

# Remove existing virtual environment if it exists
if (Test-Path "traj_opt_env") {
    Write-Host "Removing existing traj_opt_env..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force traj_opt_env
}

# Create virtual environment
Write-Host "Creating virtual environment 'traj_opt_env'..." -ForegroundColor Cyan
python -m venv traj_opt_env

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\traj_opt_env\Scripts\Activate.ps1

# Upgrade pip and setuptools
Write-Host "Upgrading pip and setuptools..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install numpy==2.2.6
pip install opencv-contrib-python==4.12.0.88
pip install tesseract-robotics==0.5.1
pip install tesseract-robotics-viewer==0.5.0
pip install aiohttp==3.13.2
pip install attrs==25.4.0
pip install typing_extensions==4.15.0

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\traj_opt_env\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nTo verify installation, run:" -ForegroundColor Yellow
Write-Host "  pip list" -ForegroundColor Cyan

