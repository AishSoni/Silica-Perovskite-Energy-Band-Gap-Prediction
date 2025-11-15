#!/usr/bin/env pwsh
# Repository Cleanup Script
# Removes redundant documentation and temporary test files

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repository Cleanup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$filesToRemove = @(
    # Redundant documentation
    "AGENTS.md",
    "ALL_ATTRIBUTES_GUIDE.md",
    "FIXES_APPLIED.md",
    "IMPLEMENTATION_SUMMARY.md",
    "improvement.md",
    "PIPELINE_QUICK_REFERENCE.md",
    "PIPELINE_REBUILD_COMPLETE.md",
    "PROJECT_README.md",
    "SUMMARY_AND_NEXT_STEPS.md",
    "SHAP_CLASSIFICATION_GUIDE.md",
    "usage.md",
    
    # Redundant test/temporary scripts
    "analyze_current_data.py",
    "check_model_performance.py",
    "download_all_data.py",
    "download_double_perovskites_main.py",
    "example_all_attributes.py",
    "quick_formula_test.py",
    "show_fixes.py",
    "test_api_connection.py",
    "test_config.py",
    "test_download_double_perovskites.py",
    "cleanup.ps1",
    
    # Logs
    "download_all_data.log"
)

$dirsToRemove = @(
    # Test output directory
    "test_output",
    
    # API docs (already in gitignore)
    "api_docs",
    
    # CatBoost temp files
    "catboost_info",
    
    # Verification scripts (one-time use)
    "verification"
)

Write-Host "Files to remove:" -ForegroundColor Yellow
foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor Gray
    }
}

Write-Host "`nDirectories to remove:" -ForegroundColor Yellow
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "  - $dir/" -ForegroundColor Gray
    }
}

Write-Host ""
$confirm = Read-Host "Do you want to proceed with cleanup? (y/N)"

if ($confirm -eq 'y' -or $confirm -eq 'Y') {
    Write-Host "`nRemoving files..." -ForegroundColor Green
    
    foreach ($file in $filesToRemove) {
        if (Test-Path $file) {
            Remove-Item $file -Force
            Write-Host "  ✓ Removed $file" -ForegroundColor Green
        }
    }
    
    Write-Host "`nRemoving directories..." -ForegroundColor Green
    foreach ($dir in $dirsToRemove) {
        if (Test-Path $dir) {
            Remove-Item $dir -Recurse -Force
            Write-Host "  ✓ Removed $dir/" -ForegroundColor Green
        }
    }
    
    Write-Host "`n✓ Cleanup complete!" -ForegroundColor Green
    Write-Host "`nRemaining important files:" -ForegroundColor Cyan
    Write-Host "  - README.md (main documentation)" -ForegroundColor Gray
    Write-Host "  - QUICK_START.md (user guide)" -ForegroundColor Gray
    Write-Host "  - requirements.txt (dependencies)" -ForegroundColor Gray
    Write-Host "  - run_pipeline.py (main pipeline)" -ForegroundColor Gray
    Write-Host "  - download_data.py (data acquisition)" -ForegroundColor Gray
    Write-Host "  - run_all.ps1 (automation)" -ForegroundColor Gray
    Write-Host "  - src/ (source code)" -ForegroundColor Gray
    Write-Host "  - paper/ (paper drafts)" -ForegroundColor Gray
    Write-Host "  - test_shap_classification.py (example script)" -ForegroundColor Gray
    
} else {
    Write-Host "`nCleanup cancelled." -ForegroundColor Yellow
}

Write-Host ""
