param(
    [string]$Version = "dev"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
$releaseStatus = 1
try {
    & $pythonExe -m tools.release.build_release --version $Version --artifact-suffix win64
    $releaseStatus = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $releaseStatus
