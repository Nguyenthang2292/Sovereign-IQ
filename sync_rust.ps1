# Sync Rust source from adaptive_trend_LTS_mini and xgboost_LTS to rust_backend
$destPath = "rust_backend\src"

Write-Host "Syncing Rust source files..." -ForegroundColor Cyan
Write-Host ""

# Source 1: adaptive_trend_LTS_mini
$source1 = "modules\adaptive_trend_LTS_mini\rust_extensions\src"
Write-Host "[1/2] Syncing from adaptive_trend_LTS_mini..." -ForegroundColor Yellow
Get-ChildItem -Path $source1 -Filter "*.rs" | Where-Object { $_.Name -ne "lib.rs" } | ForEach-Object {
    Write-Host "  Copying $($_.Name)" -ForegroundColor Gray
    Copy-Item $_.FullName -Destination $destPath -Force
}

# Source 2: xgboost_LTS
$source2 = "modules\xgboost_LTS\rust_extensions\src"
Write-Host "[2/2] Syncing from xgboost_LTS..." -ForegroundColor Yellow
Get-ChildItem -Path $source2 -Filter "*.rs" | Where-Object { $_.Name -ne "lib.rs" } | ForEach-Object {
    $newName = "xgb_$($_.Name)"
    Write-Host "  Copying $($_.Name) as $newName" -ForegroundColor Gray
    Copy-Item $_.FullName -Destination "$destPath\$newName" -Force
}

Write-Host ""
Write-Host "Sync complete!" -ForegroundColor Green
