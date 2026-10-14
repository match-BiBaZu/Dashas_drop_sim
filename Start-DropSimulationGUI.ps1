$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSCommandPath
$localEnvironment = Join-Path (Split-Path -Parent $repository) 'outputs\LocalEnvironment.ps1'
if (Test-Path -LiteralPath $localEnvironment) { . $localEnvironment }
$python = Join-Path $repository '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing local Python environment. Run 'uv sync --python 3.12 --all-extras' in $repository."
}
Start-Process -FilePath $python -ArgumentList '-m dashas_drop_sim gui' -WorkingDirectory $repository
