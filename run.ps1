<#
    Starts both halves of Mindscape and opens the browser.

    .\run.ps1              both, backend in a second window
    .\run.ps1 -NoBackend   frontend only (face channel works, no transcript)
    .\run.ps1 -Gpu         run the sidecar on CUDA instead of CPU

    Ctrl+C stops the frontend; close the other window to stop the backend.
#>
param(
    [switch]$NoBackend,
    [switch]$Gpu
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venvPython = Join-Path $root 'backend\.venv\Scripts\python.exe'

# --- checks ---------------------------------------------------------------

if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Host 'Installing frontend dependencies...' -ForegroundColor Yellow
    Push-Location (Join-Path $root 'frontend')
    npm install
    Pop-Location
}

# The camera silently fails without these, and the failure is hard to read.
if (-not (Test-Path (Join-Path $root 'frontend\public\models\tiny_face_detector_model.bin'))) {
    Write-Host 'Copying face-api weights...' -ForegroundColor Yellow
    Push-Location (Join-Path $root 'frontend')
    npm run fetch-models
    Pop-Location
}

# --- backend --------------------------------------------------------------

if (-not $NoBackend) {
    if (-not (Test-Path $venvPython)) {
        Write-Host "No virtualenv at backend\.venv - see README section 2." -ForegroundColor Red
        Write-Host "Starting the frontend alone; you'll get the face channel only." -ForegroundColor Yellow
    }
    else {
        $device = if ($Gpu) { 'cuda' } else { 'cpu' }
        Write-Host "Starting sidecar on $device (separate window)..." -ForegroundColor Cyan

        # Its own window rather than a background job, so the model-loading logs
        # and any tracebacks stay visible instead of vanishing into a buffer.
        $inner = "`$env:MINDSCAPE_DEVICE='$device'; " +
                 "Set-Location '$root\backend'; " +
                 "& '$venvPython' -m uvicorn app.main:app --port 8000"
        Start-Process powershell -ArgumentList '-NoExit', '-Command', $inner

        Write-Host 'Waiting for the sidecar...' -NoNewline
        $ready = $false
        foreach ($i in 1..40) {
            Start-Sleep -Milliseconds 500
            try {
                Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2 | Out-Null
                $ready = $true
                break
            } catch { Write-Host '.' -NoNewline }
        }
        if ($ready) {
            Write-Host ' up.' -ForegroundColor Green
            # Load the models now so the first chunk of the first recording
            # isn't swallowed by a cold start.
            try {
                Invoke-RestMethod 'http://127.0.0.1:8000/api/warmup' -Method Post -TimeoutSec 5 | Out-Null
                Write-Host 'Warming models in the background (~12s).' -ForegroundColor Cyan
            } catch { }
        }
        else {
            Write-Host ' no response - check the other window.' -ForegroundColor Red
        }
    }
}

# --- frontend -------------------------------------------------------------

Write-Host ''
Write-Host 'Mindscape -> http://localhost:5173' -ForegroundColor Green
Write-Host 'Allow camera and microphone when the browser asks.' -ForegroundColor DarkGray
Write-Host ''

Start-Process 'http://localhost:5173'

Set-Location (Join-Path $root 'frontend')
npm run dev
