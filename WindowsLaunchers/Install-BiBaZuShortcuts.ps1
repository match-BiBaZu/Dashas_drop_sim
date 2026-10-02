[CmdletBinding()]
param(
    [switch]$DesktopOnly,
    [switch]$StartMenuOnly,
    [string]$DestinationDirectory,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
if ($DesktopOnly -and $StartMenuOnly) { throw 'Choose DesktopOnly OR StartMenuOnly.' }
$launcherDirectory = Split-Path -Parent $PSCommandPath
$repository = Split-Path -Parent $launcherDirectory
$python = Join-Path $repository '.venv\Scripts\pythonw.exe'
$consolePython = Join-Path $repository '.venv\Scripts\python.exe'
$icon = Join-Path $launcherDirectory 'icons\drop-sim.ico'
$arguments = '-m dashas_drop_sim gui'

foreach ($file in @($python, $consolePython, $icon)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing: $file. Run 'uv sync --python 3.12 --all-extras' in $repository first."
    }
}
Push-Location $repository
try {
    & $consolePython -c "import PyQt6, mujoco, dashas_drop_sim.gui"
    if ($LASTEXITCODE -ne 0) { throw "Application dependencies missing. Run 'uv sync --python 3.12 --all-extras' in $repository." }
} finally { Pop-Location }
if ($CheckOnly) { Write-Host 'Launcher prerequisites OK.'; return }

$destinations = @()
if ($DestinationDirectory) { $destinations += $DestinationDirectory }
else {
    if (-not $StartMenuOnly) { $destinations += [Environment]::GetFolderPath('Desktop') }
    if (-not $DesktopOnly) { $destinations += Join-Path ([Environment]::GetFolderPath('Programs')) 'BiBaZu' }
}
$shell = New-Object -ComObject WScript.Shell
foreach ($destination in $destinations) {
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    $path = Join-Path $destination 'BiBaZu Drop Simulation.lnk'
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $python
    $shortcut.Arguments = $arguments
    $shortcut.WorkingDirectory = $repository
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = 'BiBaZu Drop Simulation (software only)'
    $shortcut.WindowStyle = 1
    $shortcut.Save()
    $saved = $shell.CreateShortcut($path)
    if ($saved.TargetPath -ne $python -or $saved.Arguments -ne $arguments -or
        $saved.IconLocation -ne "$icon,0" -or $saved.WorkingDirectory -ne $repository) {
        throw "Shortcut verification failed: $path"
    }
    Write-Host "Created: $path"
}
