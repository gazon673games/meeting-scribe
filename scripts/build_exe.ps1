param(
    [string]$Version = "dev"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
try {
    & $pythonExe scripts\build_release.py --version $Version --artifact-suffix win64
}
finally {
    Pop-Location
}
