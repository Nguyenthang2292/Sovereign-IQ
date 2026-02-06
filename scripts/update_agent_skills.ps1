# Update skills for .agent, .claude, .opencode
# Run from project root. Requires Git; HTTPS may need credentials.

$ErrorActionPreference = "Continue"
$projectRoot = $PSScriptRoot + "\.."

Write-Host "=== Update @.agent (project) ===" -ForegroundColor Cyan
$agentPath = Join-Path $projectRoot ".agent"
if (Test-Path (Join-Path $agentPath ".git")) {
    Push-Location $agentPath
    git pull
    Pop-Location
} else {
    Write-Host "  .agent is not a git repo; skip." -ForegroundColor Yellow
}

Write-Host "`n=== Update @.agent (global ~/.agent/skills) ===" -ForegroundColor Cyan
$globalAgent = Join-Path $env:USERPROFILE ".agent\skills"
if (Test-Path (Join-Path $globalAgent ".git")) {
    git -C $globalAgent pull
} else {
    Write-Host "  Path not found or not a git repo: $globalAgent" -ForegroundColor Yellow
}

Write-Host "`n=== Update @.claude (global ~/.claude/skills) ===" -ForegroundColor Cyan
$globalClaude = Join-Path $env:USERPROFILE ".claude\skills"
if (Test-Path (Join-Path $globalClaude ".git")) {
    git -C $globalClaude pull
} else {
    Write-Host "  Path not found or not a git repo: $globalClaude" -ForegroundColor Yellow
}

Write-Host "`n=== Update @.opencode (global ~/.config/opencode/skills) ===" -ForegroundColor Cyan
$globalOpenCode = Join-Path $env:USERPROFILE ".config\opencode\skills"
if (Test-Path (Join-Path $globalOpenCode ".git")) {
    git -C $globalOpenCode pull
} else {
    Write-Host "  Path not found or not a git repo: $globalOpenCode" -ForegroundColor Yellow
}

Write-Host "`nDone." -ForegroundColor Green
