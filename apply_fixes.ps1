# Apply All Fixes - Cleanup and Retrain Script
# ==============================================
# This script clears old cached data and re-runs the complete pipeline
# with all fixes applied.

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         APPLYING ALL FIXES - CLEANUP AND RETRAIN              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Cleanup old data
Write-Host "Step 1: Cleaning up old cached data..." -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

$filesToRemove = @(
    "data\processed\perovskites_features.csv",
    "data\processed\features_list.csv"
)

$foldersToClean = @(
    "models",
    "figures"
)

foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✓ Removed: $file" -ForegroundColor Green
    } else {
        Write-Host "  ○ Not found: $file" -ForegroundColor Gray
    }
}

foreach ($folder in $foldersToClean) {
    if (Test-Path $folder) {
        $count = (Get-ChildItem $folder -Recurse -File | Measure-Object).Count
        if ($count -gt 0) {
            Remove-Item "$folder\*" -Recurse -Force
            Write-Host "  ✓ Cleaned: $folder ($count files removed)" -ForegroundColor Green
        } else {
            Write-Host "  ○ Already clean: $folder" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "✓ Cleanup complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Re-run pipeline
Write-Host "Step 2: Re-running complete pipeline with fixes..." -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""
Write-Host "This will take approximately 5-10 minutes..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is active
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠️  No virtual environment detected" -ForegroundColor Yellow
    Write-Host "   Consider activating: .\perovskite\Scripts\Activate.ps1" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting pipeline..." -ForegroundColor Cyan
Write-Host ""

# Run the pipeline
python run_pipeline.py

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 3: Verify results
Write-Host "Step 3: Verifying results..." -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

if (Test-Path "data\processed\perovskites_features.csv") {
    $df = Import-Csv "data\processed\perovskites_features.csv" | Measure-Object
    $rowCount = $df.Count
    
    Write-Host "Featurized data: $rowCount rows" -ForegroundColor White
    
    if ($rowCount -lt 1500) {
        Write-Host "  ✓ Row count looks good (no more massive duplication!)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Still has $rowCount rows (expected ~986)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ Featurized data not found!" -ForegroundColor Red
}

Write-Host ""

if (Test-Path "results\all_models_summary.json") {
    $results = Get-Content "results\all_models_summary.json" | ConvertFrom-Json
    
    if ($results.'LightGBM Regression (All, Mean)') {
        $r2 = $results.'LightGBM Regression (All, Mean)'.'R²'
        $mae = $results.'LightGBM Regression (All, Mean)'.'MAE'
        
        Write-Host "Model Performance:" -ForegroundColor White
        Write-Host "  R²: $([math]::Round($r2, 3))" -ForegroundColor White
        Write-Host "  MAE: $([math]::Round($mae, 3)) eV" -ForegroundColor White
        
        if ($r2 -gt 0.30) {
            Write-Host "  ✓ Good improvement in R²!" -ForegroundColor Green
        } elseif ($r2 -gt 0.20) {
            Write-Host "  ○ Moderate improvement" -ForegroundColor Yellow
        } else {
            Write-Host "  ⚠️  Still low R²" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ✗ Results not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 4: Test SHAP fix
Write-Host "Step 4: Testing SHAP fix..." -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

if (Test-Path "models\xgb_classification.pkl") {
    Write-Host "Running SHAP test..." -ForegroundColor Cyan
    python test_shap_fix.py
} else {
    Write-Host "  ⚠️  XGBoost model not found, skipping SHAP test" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ DONE! Check the outputs in:" -ForegroundColor Green
Write-Host "   • figures/             - Updated plots" -ForegroundColor White
Write-Host "   • results/             - Updated metrics" -ForegroundColor White
Write-Host "   • models/              - Retrained models" -ForegroundColor White
Write-Host ""
Write-Host "For detailed analysis, see:" -ForegroundColor Cyan
Write-Host "   • results/training_challenges_and_solutions.md  - Research paper documentation" -ForegroundColor White
Write-Host ""
