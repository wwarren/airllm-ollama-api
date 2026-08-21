#!/usr/bin/env bash
#
# Install the airllm-ollama-api service: an Ollama-compatible HTTP API
# served by AirLLM.
#
# Run from the repository root as your normal user (not root):
#     ./airllm-ollama-api-install.sh
#
# Re-running is safe: it refreshes the code and dependencies and leaves an
# existing .env alone.

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/airllm-ollama-api}"
SERVICE_NAME="${SERVICE_NAME:-airllm-ollama-api}"
SERVICE_USER="${SERVICE_USER:-$(id -un)}"
SERVICE_GROUP="${SERVICE_GROUP:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
APP_ENTRYPOINT="airllm_ollama_api.py"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n[%s/%s] %s\n' "$1" "$TOTAL_STEPS" "$2"; }
read_env_value() {
  local file="$1"
  local key="$2"
  [[ -f "$file" ]] || return 1
  sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*[\"']?([^\"'#[:space:]]+)[\"']?.*$/\1/p" "$file" | tail -1
}
validate_name() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[A-Za-z0-9_.@-]+$ ]] || die "$label contains invalid characters: $value"
}
validate_app_dir() {
  local resolved
  [[ -n "$APP_DIR" && "$APP_DIR" = /* ]] || die "APP_DIR must be an absolute path"
  resolved="$(readlink -m "$APP_DIR")"
  case "$resolved" in
    /|/opt|/usr|/etc|/var|/home|/boot|/efi|/srv)
      die "refusing unsafe APP_DIR: $resolved"
      ;;
    /usr/*|/etc/*|/boot/*|/efi/*)
      # The unit sets ProtectSystem=full, which mounts /usr, /boot, /efi and
      # /etc read-only for the service. Installing there succeeds and then
      # fails at runtime when the model cache is written.
      die "APP_DIR is read-only for the service under ProtectSystem=full: $resolved"
      ;;
  esac
  APP_DIR="$resolved"
}
validate_port() {
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "AIRLLM_PORT must be numeric, got: $PORT"
  (( PORT >= 1 && PORT <= 65535 )) || die "AIRLLM_PORT must be between 1 and 65535, got: $PORT"
}
port_is_listening() {
  command -v ss >/dev/null 2>&1 || return 1
  ss -H -ltn "( sport = :$PORT )" 2>/dev/null | grep -q .
}
TOTAL_STEPS=6

trap 'die "failed on line $LINENO"' ERR

# --- preflight -------------------------------------------------------------

if [[ $EUID -eq 0 ]]; then
  die "run as your normal user, not root — the script calls sudo where it needs to"
fi

command -v sudo >/dev/null 2>&1 || die "sudo is required"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found"
"$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1 \
  || die "the venv module is missing — install python3-venv (apt install python3-venv)"
command -v readlink >/dev/null 2>&1 || die "readlink is required"
command -v sed >/dev/null 2>&1 || die "sed is required"

validate_name "SERVICE_NAME" "$SERVICE_NAME"
validate_name "SERVICE_USER" "$SERVICE_USER"
id -u "$SERVICE_USER" >/dev/null 2>&1 || die "SERVICE_USER does not exist: $SERVICE_USER"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn "$SERVICE_USER")}"
validate_name "SERVICE_GROUP" "$SERVICE_GROUP"
validate_app_dir
sudo -v

if ! command -v systemctl >/dev/null 2>&1 || [[ ! -d /run/systemd/system ]]; then
  die "systemd is not running. On WSL, enable it with 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'."
fi

for required in "$APP_ENTRYPOINT" requirements.txt env.example; do
  [[ -f "$REPO_DIR/$required" ]] || die "$required not found next to airllm-ollama-api-install.sh"
done

# --- 1. directories --------------------------------------------------------

step 1 "Creating $APP_DIR"
sudo mkdir -p "$APP_DIR"
sudo chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

# --- 2. application files --------------------------------------------------

step 2 "Copying application files"
install -m 0644 "$REPO_DIR/$APP_ENTRYPOINT" "$APP_DIR/$APP_ENTRYPOINT"
install -m 0644 "$REPO_DIR/requirements.txt" "$APP_DIR/requirements.txt"
install -m 0644 "$REPO_DIR/env.example" "$APP_DIR/env.example"
if [[ -f "$REPO_DIR/README.md" ]]; then
  install -m 0644 "$REPO_DIR/README.md" "$APP_DIR/README.md"
fi

if [[ -f "$APP_DIR/.env" ]]; then
  echo "  keeping existing $APP_DIR/.env"
elif [[ -f "$REPO_DIR/.env" ]]; then
  install -m 0600 "$REPO_DIR/.env" "$APP_DIR/.env"
  echo "  copied .env from the repo"
else
  install -m 0600 "$REPO_DIR/env.example" "$APP_DIR/.env"
  sed -i -E "s|^HF_HOME=.*|HF_HOME=${APP_DIR}/hf-cache|" "$APP_DIR/.env"
  echo "  created $APP_DIR/.env from the example — edit it before serving a real model"
fi

PORT="$(read_env_value "$APP_DIR/.env" AIRLLM_PORT || true)"
PORT="${PORT:-11434}"
validate_port

if [[ "$PORT" == "11434" ]] && systemctl is-active --quiet ollama.service 2>/dev/null; then
  echo "WARNING: ollama.service is running and may own port 11434."
  echo "         Stop it (sudo systemctl stop ollama) or set AIRLLM_PORT to something else in $APP_DIR/.env."
elif port_is_listening && ! systemctl is-active --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
  echo "WARNING: port $PORT is already listening before ${SERVICE_NAME}.service starts."
  echo "         If startup fails, check the owning process with: sudo ss -ltnp 'sport = :$PORT'"
fi

echo "Installing to : $APP_DIR"
echo "Service name  : ${SERVICE_NAME}.service"
echo "Running as    : ${SERVICE_USER}:${SERVICE_GROUP}"
echo "Port          : $PORT"

# --- 3. virtualenv ---------------------------------------------------------

step 3 "Creating the virtualenv and installing dependencies (this takes a while)"
if [[ ! -x "$APP_DIR/venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel >/dev/null
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# --- 4. import check -------------------------------------------------------

step 4 "Verifying the install"
"$APP_DIR/venv/bin/python" - <<'PYCHECK'
import importlib.util
missing = [m for m in ("airllm", "transformers", "torch", "fastapi", "uvicorn")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing modules after install: " + ", ".join(missing))
print("  all required modules import cleanly")
PYCHECK
"$APP_DIR/venv/bin/python" -m py_compile "$APP_DIR/$APP_ENTRYPOINT"

# --- 5. systemd unit -------------------------------------------------------

step 5 "Writing /etc/systemd/system/${SERVICE_NAME}.service"
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<SERVICE_EOF
[Unit]
Description=airllm-ollama-api — Ollama-compatible HTTP API served by AirLLM on port ${PORT}
Documentation=file://${APP_DIR}/README.md
After=network-online.target
Wants=network-online.target
# Stop the restart loop instead of re-downloading weights forever on a
# misconfiguration (bad model id, missing HF_TOKEN).
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/${APP_ENTRYPOINT}
Restart=on-failure
RestartSec=15
# Loading a 70B can take a very long time; never kill it mid-shard.
TimeoutStopSec=120
StandardOutput=journal
StandardError=journal

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectControlGroups=true
ProtectKernelTunables=true

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# --- 6. start --------------------------------------------------------------

step 6 "Enabling and starting the service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
sudo systemctl restart "${SERVICE_NAME}.service"

sleep 3
if ! systemctl is-active --quiet "${SERVICE_NAME}.service"; then
  echo
  echo "The service is not running. Last 40 log lines:"
  sudo journalctl -u "${SERVICE_NAME}.service" -n 40 --no-pager || true
  exit 1
fi

cat <<DONE

=========================================================
${SERVICE_NAME}.service is running on port ${PORT}
(Ollama-compatible HTTP API served by AirLLM).

The model loads in the background — the API answers
/api/version and /api/tags immediately, and returns 503 on
/api/chat until the weights are ready.

  Load status : curl -s localhost:${PORT}/health
  Live logs   : journalctl -u ${SERVICE_NAME}.service -f
  Smoke test  : curl -s localhost:${PORT}/api/chat -d '{
                  "model":"local","messages":[{"role":"user","content":"hi"}]
                }'

Config lives in ${APP_DIR}/.env — edit it and run
'sudo systemctl restart ${SERVICE_NAME}' to apply changes.
=========================================================
DONE
