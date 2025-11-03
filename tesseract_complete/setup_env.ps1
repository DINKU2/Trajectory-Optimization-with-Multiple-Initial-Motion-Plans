# Setup script for Tesseract environment variables
# Run this with: . .\setup_env.ps1
# This will set the environment variables for the current PowerShell session

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TESSERACT_RESOURCE_PATH = Join-Path $scriptPath "tesseract"
$env:TESSERACT_TASK_COMPOSER_CONFIG_FILE = Join-Path $scriptPath "tesseract_planning\tesseract_task_composer\config\task_composer_plugins_no_trajopt_ifopt.yaml"

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  TESSERACT_RESOURCE_PATH = $env:TESSERACT_RESOURCE_PATH"
Write-Host "  TESSERACT_TASK_COMPOSER_CONFIG_FILE = $env:TESSERACT_TASK_COMPOSER_CONFIG_FILE"


