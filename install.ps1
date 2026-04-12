# ============================================================
#  Vigilante - One-Line Installer (Windows PowerShell)
#  Usage (paste into PowerShell):
#    irm https://raw.githubusercontent.com/2besh2002-pixel/vigilante-CopyBashayer/main/install.ps1 | iex
# ============================================================

$ErrorActionPreference = "Stop"

function Ok($msg)   { Write-Host "  $([char]0x2714)  $msg" -ForegroundColor Green }
function Info($msg) { Write-Host "  ->  $msg" -ForegroundColor Yellow }
function Err($msg)  { Write-Host "`n  X  ERROR: $msg`n" -ForegroundColor Red; exit 1 }
function Step($n,$msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

Write-Host "`n  Vigilante Security - Installer`n" -ForegroundColor Cyan

# ── 1. Python ──────────────────────────────────────────────
Step "1/3" "Checking Python..."

$python = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd -c "import sys; print(sys.version_info >= (3,8))" 2>$null
        if ($ver -eq "True") { $python = $cmd; break }
    } catch {}
}

if (-not $python) {
    Err "Python 3.8+ is required.`n`n  Download from: https://www.python.org/downloads/`n  IMPORTANT: Check 'Add Python to PATH' during install."
}
Ok "$(& $python --version)"

# ── 2. pipx ────────────────────────────────────────────────
Step "2/3" "Setting up pipx..."

$pipxOk = $false
try { pipx --version | Out-Null; $pipxOk = $true } catch {}

if (-not $pipxOk) {
    Info "pipx not found — installing..."
    & $python -m pip install --user --quiet pipx
    & $python -m pipx ensurepath --force | Out-Null
    # Refresh PATH for this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" + $env:PATH
    Info "pipx installed."
}
Ok "pipx ready."

# ── 3. Install Vigilante ───────────────────────────────────
Step "3/3" "Installing Vigilante via pipx..."

$pipxCmd = if (Get-Command pipx -ErrorAction SilentlyContinue) { "pipx" } else { "$python -m pipx" }
Invoke-Expression "$pipxCmd install git+https://github.com/2besh2002-pixel/vigilante-CopyBashayer --force"

# ── Write config ───────────────────────────────────────────
$configDir = "$env:USERPROFILE\.vigilante"
$modelsDir  = "$configDir\models"
$envFile    = "$configDir\.env"

New-Item -ItemType Directory -Force -Path $configDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

if (-not (Test-Path $envFile)) {
    @"
DATABASE_URL=postgresql://neondb_owner:npg_xwSq6emIHk2v@ep-jolly-hall-abac7zg7-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
RESEND_API_KEY=re_K6L2ohfP_EN3BDtPaKCQ9yS9mco6hX6QQ
SESSION_TIMEOUT_HOURS=24
OTP_EXPIRY_MINUTES=10
MAX_LOGIN_ATTEMPTS=5
DEFAULT_MODEL_DIR=$modelsDir
DEFAULT_THRESHOLD=0.8
DEFAULT_EPOCHS=50
DEFAULT_LEARNING_RATE=0.001
LOG_LEVEL=INFO
LOG_FILE=$configDir\vigilante.log
AUDIT_LOG_RETENTION_DAYS=90
"@ | Set-Content $envFile
}

# ── Done ───────────────────────────────────────────────────
Write-Host "`n  Done! Vigilante is installed.`n" -ForegroundColor Green
Write-Host "  vigilante login      - get started" -ForegroundColor White
Write-Host "  vigilante --help     - see all commands" -ForegroundColor White
Write-Host "`n  Tip: If 'vigilante' is not found, open a new terminal.`n" -ForegroundColor Yellow
