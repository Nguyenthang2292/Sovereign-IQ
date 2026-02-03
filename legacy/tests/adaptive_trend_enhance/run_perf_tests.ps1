#!/usr/bin/env pwsh
# Quick Performance Test Runner for PowerShell
# Usage: .\run_perf_tests.ps1 [mode] [suite]
#   mode: fast (default) | full | ci
#   suite: all (default) | regression | comparison

param(
    [ValidateSet('fast', 'full', 'ci')]
    [string]$Mode = 'fast',

    [ValidateSet('all', 'regression', 'comparison')]
    [string]$Suite = 'all'
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Performance Test Runner" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Determine test files to run
$TestFiles = switch ($Suite) {
    'all' {
        "tests/adaptive_trend_enhance/test_performance_regression.py tests/adaptive_trend_enhance/test_performance.py"
    }
    'regression' {
        "tests/adaptive_trend_enhance/test_performance_regression.py"
    }
    'comparison' {
        "tests/adaptive_trend_enhance/test_performance.py"
    }
}

Write-Host "Suite: $($Suite.ToUpper())" -ForegroundColor Magenta
Write-Host ""

switch ($Mode) {
    'fast' {
        Write-Host "Mode: FAST DEVELOPMENT" -ForegroundColor Green
        Write-Host "- Iterations: 3"
        Write-Host "- Skip slow tests"
        Write-Host "- Single threaded"
        Write-Host ""
        pytest $TestFiles -n 0 -m "not slow" -v
    }
    'full' {
        Write-Host "Mode: FULL TEST SUITE" -ForegroundColor Yellow
        Write-Host "- Iterations: 5"
        Write-Host "- All tests"
        Write-Host "- Single threaded"
        Write-Host ""
        $env:PERF_ITERATIONS = "5"
        pytest $TestFiles -n 0 -m performance -v
    }
    'ci' {
        Write-Host "Mode: CI/PRODUCTION" -ForegroundColor Red
        Write-Host "- Iterations: 10"
        Write-Host "- All tests with coverage"
        Write-Host "- Single threaded"
        Write-Host ""
        $env:PERF_ITERATIONS = "10"
        pytest $TestFiles -n 0 -m performance --cov=modules.adaptive_trend_enhance --cov-report=html -v
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Tests completed!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
