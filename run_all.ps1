# run_all.ps1
# Complete pipeline script for Perovskite ML project
# Runs data loading, feature engineering, preprocessing, training, and evaluation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Perovskite Bandgap Prediction Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".\perovskite"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv perovskite
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "$venvPath\Scripts\Activate.ps1"

# Install/upgrade dependencies
Write-Host ""
Write-Host "Installing/upgrading dependencies..." -ForegroundColor Green
#pip install --upgrade pip
#pip install -r requirements.txt
#not needed everytime as dependencies have been taken care of

# Check Python version
Write-Host ""
Write-Host "Python version:" -ForegroundColor Cyan
python --version

# Run the main pipeline
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running ML Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python run_pipeline.py

# Check if pipeline succeeded
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Check the following directories for outputs:" -ForegroundColor Yellow
    Write-Host "  - data/processed/  : Processed datasets" -ForegroundColor White
    Write-Host "  - models/          : Trained models" -ForegroundColor White
    Write-Host "  - figures/         : Plots and visualizations" -ForegroundColor White
    Write-Host "  - results/         : Evaluation metrics" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Pipeline failed with errors" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
