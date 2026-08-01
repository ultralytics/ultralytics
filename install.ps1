# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Install the Ultralytics `yolo` command, including Python itself if it is missing, using uv https://docs.astral.sh/uv/
# Example usage: powershell -c "irm https://raw.githubusercontent.com/ultralytics/ultralytics/main/install.ps1 | iex"
# Options: $env:ULTRALYTICS_PYTHON sets the Python version, $env:ULTRALYTICS_PACKAGE sets the PyPI package

$ErrorActionPreference = "Stop"

$pythonVersion = if ($env:ULTRALYTICS_PYTHON) { $env:ULTRALYTICS_PYTHON } else { "3.12" }
$package = if ($env:ULTRALYTICS_PACKAGE) { $env:ULTRALYTICS_PACKAGE } else { "ultralytics" }
$binDir = "$env:USERPROFILE\.local\bin"

function Info($message) {
    Write-Host "Ultralytics " -ForegroundColor Blue -NoNewline
    Write-Host $message
}

# uv downloads and manages Python, so no system Python is required
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Info "installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$binDir;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was installed but is not on PATH, restart your terminal and rerun"
    }
}

Info "installing $package on Python $pythonVersion, this may take a few minutes..."
uv tool install --upgrade --python-preference only-managed --python $pythonVersion $package
uv tool update-shell # add the tool directory to PATH for new terminals
$env:Path = "$(uv tool dir --bin);$env:Path"

yolo version
Info "run 'yolo predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg' to get started"
Info "open a new terminal if the 'yolo' command is not found"
