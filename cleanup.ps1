# Cleanup Script for Perovskite Project
# ======================================
# Removes generated data, models, and figures to prepare for a fresh pipeline run
# Run this between pipeline executions to ensure clean results

param(
    [switch]$KeepRaw,
    [switch]$KeepModels,
    [switch]$KeepFigures,
    [switch]$KeepResults,
    [switch]$All,
    [switch]$DryRun
)

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            CLEANUP SCRIPT - PEROVSKITE PROJECT                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No files will be deleted" -ForegroundColor Yellow
    Write-Host ""
}

$totalSize = 0
$totalFiles = 0

function Remove-FilesInDirectory {
    param(
        [string]$Path,
        [string]$Pattern = "*",
        [string]$Description,
        [switch]$Recurse,
        [switch]$Skip
    )
    
    if ($Skip) {
        Write-Host "  ⊘ Skipping: $Description" -ForegroundColor Gray
        return
    }
    
    if (-not (Test-Path $Path)) {
        Write-Host "  ○ Not found: $Description" -ForegroundColor DarkGray
        return
    }
    
    $files = Get-ChildItem -Path $Path -Filter $Pattern -Recurse:$Recurse -File -ErrorAction SilentlyContinue
    $count = ($files | Measure-Object).Count
    
    if ($count -eq 0) {
        Write-Host "  ○ Already clean: $Description" -ForegroundColor DarkGray
        return
    }
    
    $size = ($files | Measure-Object -Property Length -Sum).Sum
    $sizeKB = [math]::Round($size / 1KB, 2)
    $sizeMB = [math]::Round($size / 1MB, 2)
    
    if ($DryRun) {
        Write-Host "  ◉ Would remove: $count files ($sizeMB MB) from $Description" -ForegroundColor Yellow
    } else {
        $files | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ Removed: $count files ($sizeMB MB) from $Description" -ForegroundColor Green
    }
    
    $script:totalFiles += $count
    $script:totalSize += $size
}

# Display options
Write-Host "Options:" -ForegroundColor Cyan
if ($All) {
    Write-Host "  • Removing ALL generated files (raw data, processed data, models, figures, results)" -ForegroundColor White
} else {
    Write-Host "  • Keeping raw data: $($KeepRaw -or -not $All)" -ForegroundColor White
    Write-Host "  • Keeping models: $KeepModels" -ForegroundColor White
    Write-Host "  • Keeping figures: $KeepFigures" -ForegroundColor White
    Write-Host "  • Keeping results: $KeepResults" -ForegroundColor White
}
Write-Host ""

# 1. Processed Data (always clean unless -All)
Write-Host "1. Cleaning processed data..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "data\processed" -Pattern "*.csv" -Description "Processed CSV files"
Remove-FilesInDirectory -Path "data\processed" -Pattern "*.pkl" -Description "Processed pickle files"
Remove-FilesInDirectory -Path "data\processed" -Pattern "*.joblib" -Description "Processed joblib files"
Remove-FilesInDirectory -Path "data\processed" -Pattern "*.txt" -Description "Feature name files"
Remove-FilesInDirectory -Path "data\processed" -Pattern "*.json" -Description "Preprocessing metadata"
Write-Host ""

# 2. Raw Data (only if -All specified)
if ($All -and -not $KeepRaw) {
    Write-Host "2. Cleaning raw data..." -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Remove-FilesInDirectory -Path "data\raw" -Pattern "*.csv" -Description "Raw CSV files"
    Remove-FilesInDirectory -Path "data\raw" -Pattern "*.json" -Description "Raw metadata files"
    Write-Host ""
}

# 3. Models
Write-Host "3. Cleaning trained models..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "models" -Pattern "*.pkl" -Description "Model pickle files" -Skip:$KeepModels
Remove-FilesInDirectory -Path "models" -Pattern "*.joblib" -Description "Model joblib files" -Skip:$KeepModels
Remove-FilesInDirectory -Path "models" -Pattern "*.h5" -Description "Keras models" -Skip:$KeepModels
Remove-FilesInDirectory -Path "models" -Pattern "*.pt" -Description "PyTorch models" -Skip:$KeepModels
Remove-FilesInDirectory -Path "models" -Pattern "*.pth" -Description "PyTorch models" -Skip:$KeepModels
Write-Host ""

# 4. Figures
Write-Host "4. Cleaning figures..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "figures" -Pattern "*.png" -Description "PNG figures" -Recurse -Skip:$KeepFigures
Remove-FilesInDirectory -Path "figures" -Pattern "*.jpg" -Description "JPG figures" -Recurse -Skip:$KeepFigures
Remove-FilesInDirectory -Path "figures" -Pattern "*.svg" -Description "SVG figures" -Recurse -Skip:$KeepFigures
Remove-FilesInDirectory -Path "figures" -Pattern "*.pdf" -Description "PDF figures" -Recurse -Skip:$KeepFigures
Write-Host ""

# 5. Results
Write-Host "5. Cleaning results..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "results" -Pattern "*.csv" -Description "Result CSV files" -Skip:$KeepResults
Remove-FilesInDirectory -Path "results" -Pattern "*.txt" -Description "Result text files" -Skip:$KeepResults
Remove-FilesInDirectory -Path "results" -Pattern "all_models_summary.json" -Description "Model summary JSON" -Skip:$KeepResults
Write-Host ""

# 6. CatBoost logs
Write-Host "6. Cleaning CatBoost logs..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "catboost_info" -Pattern "*" -Description "CatBoost training logs" -Recurse
Write-Host ""

# 7. Python cache
Write-Host "7. Cleaning Python cache..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
$pycacheDirs = Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    $files = Get-ChildItem -Path $dir.FullName -File -Recurse
    $count = ($files | Measure-Object).Count
    if ($count -gt 0) {
        if ($DryRun) {
            Write-Host "  ◉ Would remove: $count files from $($dir.FullName)" -ForegroundColor Yellow
        } else {
            Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  ✓ Removed: $($dir.FullName)" -ForegroundColor Green
        }
        $script:totalFiles += $count
    }
}
Write-Host ""

# 8. Temporary files
Write-Host "8. Cleaning temporary files..." -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Remove-FilesInDirectory -Path "." -Pattern "*.tmp" -Description "Temp files"
Remove-FilesInDirectory -Path "." -Pattern "*.temp" -Description "Temp files"
Remove-FilesInDirectory -Path "." -Pattern "*.bak" -Description "Backup files"
Remove-FilesInDirectory -Path "." -Pattern "test_shap_fix.png" -Description "Test files"
Write-Host ""

# Summary
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
$totalSizeMB = [math]::Round($totalSize / 1MB, 2)
$totalSizeGB = [math]::Round($totalSize / 1GB, 3)

if ($DryRun) {
    Write-Host "DRY RUN SUMMARY:" -ForegroundColor Yellow
    Write-Host "  Would remove: $totalFiles files" -ForegroundColor White
    Write-Host "  Would free: $totalSizeMB MB ($totalSizeGB GB)" -ForegroundColor White
    Write-Host ""
    Write-Host "Run without -DryRun to actually delete files" -ForegroundColor Cyan
} else {
    Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
    Write-Host "  Removed: $totalFiles files" -ForegroundColor White
    Write-Host "  Freed: $totalSizeMB MB ($totalSizeGB GB)" -ForegroundColor White
}

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Usage examples
Write-Host "💡 Usage examples:" -ForegroundColor Cyan
Write-Host "  .\cleanup.ps1                    # Standard cleanup (keeps raw data)" -ForegroundColor Gray
Write-Host "  .\cleanup.ps1 -All               # Remove everything including raw data" -ForegroundColor Gray
Write-Host "  .\cleanup.ps1 -KeepModels        # Clean but keep trained models" -ForegroundColor Gray
Write-Host "  .\cleanup.ps1 -KeepFigures       # Clean but keep figures" -ForegroundColor Gray
Write-Host "  .\cleanup.ps1 -DryRun            # Preview what would be deleted" -ForegroundColor Gray
Write-Host "  .\cleanup.ps1 -All -DryRun       # Preview full cleanup" -ForegroundColor Gray
Write-Host ""
