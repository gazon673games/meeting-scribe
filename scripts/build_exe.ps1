param(
    [string]$Version = "dev"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$specPath = Join-Path $repoRoot "packaging\meeting_scribe.spec"
$distDir = Join-Path $repoRoot "dist\meeting-scribe"
$exePath = Join-Path $distDir "meeting-scribe.exe"
$safeVersion = [regex]::Replace($Version, "[^A-Za-z0-9_.-]+", "-")
if ([string]::IsNullOrWhiteSpace($safeVersion)) {
    $safeVersion = "manual"
}
$zipPath = Join-Path $repoRoot ("dist\meeting-scribe-{0}-win64.zip" -f $safeVersion)
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
try {
    & $pythonExe -m PyInstaller --noconfirm --clean $specPath

    if (-not (Test-Path -LiteralPath $distDir)) {
        throw "Expected PyInstaller output was not found: $distDir"
    }
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Expected executable was not found: $exePath"
    }

    $smoke = Start-Process -FilePath $exePath -ArgumentList "--smoke-import" -PassThru -Wait
    if ($smoke.ExitCode -ne 0) {
        throw "Packaged executable smoke check failed with exit code $($smoke.ExitCode)"
    }

    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipPath -Force
    Write-Host "Built $zipPath"
}
finally {
    Pop-Location
}
