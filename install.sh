#!/usr/bin/env bash
# ============================================================
#  Vigilante - One-Line Installer (macOS & Linux)
#  Usage: curl -fsSL https://raw.githubusercontent.com/AljawharaK/vigilante/main/install.sh | bash
# ============================================================

set -e

BOLD="\033[1m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
RED="\033[0;31m"; CYAN="\033[0;36m"; RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✔${RESET}  $1"; }
info() { echo -e "  ${YELLOW}→${RESET}  $1"; }
err()  { echo -e "\n  ${RED}✘  ERROR:${RESET} $1\n"; exit 1; }
step() { echo -e "\n${CYAN}${BOLD}[$1]${RESET} $2"; }

echo -e "\n${CYAN}${BOLD}  Vigilante Security — Installer${RESET}\n"

# ── 1. Python ──────────────────────────────────────────────
step "1/3" "Checking Python..."

PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(sys.version_info >= (3,8))" 2>/dev/null)
    if [ "$VER" = "True" ]; then PYTHON="$cmd"; break; fi
  fi
done

if [ -z "$PYTHON" ]; then
  err "Python 3.8+ is required but was not found.\n\n  macOS:         brew install python\n  Ubuntu/Debian: sudo apt install python3\n  Download:      https://www.python.org/downloads/"
fi
ok "$(${PYTHON} --version)"

# ── 2. pipx ────────────────────────────────────────────────
step "2/3" "Setting up pipx..."

if ! command -v pipx &>/dev/null; then
  info "pipx not found — installing..."
  if [ "$(uname -s)" = "Darwin" ] && command -v brew &>/dev/null; then
    brew install pipx &>/dev/null
  else
    "$PYTHON" -m pip install --user --quiet pipx
  fi
  "$PYTHON" -m pipx ensurepath --force &>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
  info "pipx installed and added to PATH."
fi
ok "pipx ready."

# ── 3. Install Vigilante ───────────────────────────────────
step "3/3" "Installing Vigilante via pipx..."

PIPX_CMD="pipx"; command -v pipx &>/dev/null || PIPX_CMD="$PYTHON -m pipx"
$PIPX_CMD install git+https://github.com/AljawharaK/vigilante --force

# ── Write config ───────────────────────────────────────────
CONFIG_DIR="$HOME/.vigilante"
mkdir -p "$CONFIG_DIR/models"

if [ ! -f "$CONFIG_DIR/.env" ]; then
  cat > "$CONFIG_DIR/.env" <<EOF
DATABASE_URL=postgresql://neondb_owner:npg_xwSq6emIHk2v@ep-jolly-hall-abac7zg7-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
RESEND_API_KEY=re_K6L2ohfP_EN3BDtPaKCQ9yS9mco6hX6QQ
SESSION_TIMEOUT_HOURS=24
OTP_EXPIRY_MINUTES=10
MAX_LOGIN_ATTEMPTS=5
DEFAULT_MODEL_DIR=$CONFIG_DIR/models
DEFAULT_THRESHOLD=0.8
DEFAULT_EPOCHS=50
DEFAULT_LEARNING_RATE=0.001
LOG_LEVEL=INFO
LOG_FILE=$CONFIG_DIR/vigilante.log
AUDIT_LOG_RETENTION_DAYS=90
EOF
fi

# ── Done ───────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}  ✔ Done! Vigilante is installed.${RESET}\n"
echo -e "  ${BOLD}vigilante login${RESET}     — get started"
echo -e "  ${BOLD}vigilante --help${RESET}    — see all commands"
echo -e "\n  ${YELLOW}Tip: If 'vigilante' is not found, open a new terminal.${RESET}\n"
