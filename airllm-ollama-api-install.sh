#!/usr/bin/env bash
#
# Install the airllm-ollama-api service: an Ollama-compatible HTTP API
# served by AirLLM.
#
# Run from the repository root, either as a normal user with sudo or as root
# (e.g. inside a container):
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
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-auto}"
INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES:-auto}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
# Run a command with privilege: directly when already root (containers),
# through sudo otherwise.
as_root() {
  if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}
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
nvidia_gpu_detected() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}
install_cuda_torch_if_available() {
  case "$INSTALL_CUDA_TORCH" in
    auto|1|true|yes) ;;
    0|false|no)
      echo "  skipping CUDA torch install because INSTALL_CUDA_TORCH=$INSTALL_CUDA_TORCH"
      return
      ;;
    *)
      die "INSTALL_CUDA_TORCH must be auto, 1, or 0; got: $INSTALL_CUDA_TORCH"
      ;;
  esac

  if ! nvidia_gpu_detected; then
    echo "  no NVIDIA GPU detected; keeping the torch package from requirements.txt"
    return
  fi

  echo "  NVIDIA GPU detected:"
  nvidia-smi -L | sed 's/^/    /'
  echo "  installing CUDA-enabled torch from $TORCH_CUDA_INDEX_URL"
  "$APP_DIR/venv/bin/pip" install --upgrade torch --index-url "$TORCH_CUDA_INDEX_URL"
  "$APP_DIR/venv/bin/python" - <<'PYCHECK'
import torch
print("  torch:", torch.__version__)
print("  cuda build:", torch.version.cuda)
print("  cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA torch installed, but torch.cuda.is_available() is false")
PYCHECK
}
install_bitsandbytes_if_requested() {
  case "$INSTALL_BITSANDBYTES" in
    auto|1|true|yes) ;;
    0|false|no)
      echo "  skipping bitsandbytes install because INSTALL_BITSANDBYTES=$INSTALL_BITSANDBYTES"
      return
      ;;
    *)
      die "INSTALL_BITSANDBYTES must be auto, 1, or 0; got: $INSTALL_BITSANDBYTES"
      ;;
  esac

  if [[ "$INSTALL_BITSANDBYTES" == "auto" ]] && ! nvidia_gpu_detected; then
    echo "  no NVIDIA GPU detected; skipping bitsandbytes in auto mode"
    return
  fi

  echo "  installing bitsandbytes"
  "$APP_DIR/venv/bin/pip" install --upgrade bitsandbytes
  "$APP_DIR/venv/bin/python" - <<'PYCHECK'
import importlib.metadata
import bitsandbytes
print("  bitsandbytes:", importlib.metadata.version("bitsandbytes"))
PYCHECK
}
TOTAL_STEPS=7

trap 'die "failed on line $LINENO"' ERR

# --- preflight -------------------------------------------------------------

if [[ $EUID -eq 0 ]]; then
  RUNNING_AS_ROOT=1
  SUDO_HINT=""
else
  RUNNING_AS_ROOT=0
  SUDO_HINT="sudo "
  command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root"
fi
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

if (( RUNNING_AS_ROOT == 0 )); then
  sudo -v || die "sudo authentication failed"
fi
if [[ "$SERVICE_USER" == "root" ]]; then
  echo "WARNING: the service will run as root. It has no authentication, so keep"
  echo "         AIRLLM_HOST=127.0.0.1 or firewall the port. To use a dedicated"
  echo "         account instead: SERVICE_USER=airllm ./airllm-ollama-api-install.sh"
fi

if ! command -v systemctl >/dev/null 2>&1 || [[ ! -d /run/systemd/system ]]; then
  die "systemd is not running. On WSL, enable it with 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'."
fi

for required in "$APP_ENTRYPOINT" requirements.txt env.example; do
  [[ -f "$REPO_DIR/$required" ]] || die "$required not found next to airllm-ollama-api-install.sh"
done

# --- 1. directories --------------------------------------------------------

step 1 "Creating $APP_DIR"
as_root mkdir -p "$APP_DIR"
as_root chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

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
  if nvidia_gpu_detected; then
    sed -i -E "s|^[#[:space:]]*AIRLLM_DEVICE=.*|AIRLLM_DEVICE=cuda:0|" "$APP_DIR/.env"
  fi
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
install_cuda_torch_if_available
install_bitsandbytes_if_requested

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
as_root tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<SERVICE_EOF
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
as_root systemctl daemon-reload
as_root systemctl enable "${SERVICE_NAME}.service"
as_root systemctl restart "${SERVICE_NAME}.service"

# --- 7. verify -------------------------------------------------------------

step 7 "Verifying the service"

HOST="$(read_env_value "$APP_DIR/.env" AIRLLM_HOST || true)"
PROBE_HOST="${HOST:-127.0.0.1}"
if [[ "$PROBE_HOST" == "0.0.0.0" || "$PROBE_HOST" == "::" ]]; then
  PROBE_HOST=127.0.0.1
fi

if (( RUNNING_AS_ROOT == 1 )); then
  JOURNAL="journalctl -u ${SERVICE_NAME}.service"
else
  JOURNAL="sudo journalctl -u ${SERVICE_NAME}.service"
fi

show_failure() {
  echo
  echo "--- systemctl status ${SERVICE_NAME} ---"
  as_root systemctl status "${SERVICE_NAME}.service" --no-pager --lines=0 || true
  echo
  echo "--- last 40 log lines ---"
  as_root journalctl -u "${SERVICE_NAME}.service" -n 40 --no-pager || true
  echo
  echo "Follow the log with: ${JOURNAL} -f"
}

sleep 3
if ! systemctl is-active --quiet "${SERVICE_NAME}.service"; then
  echo "  ${SERVICE_NAME}.service is NOT active."
  show_failure
  exit 1
fi
echo "  unit is active"

# The HTTP port binds before the model finishes loading, so /api/version
# answers almost immediately. Anything slower than this means real trouble.
probe() {
  "$APP_DIR/venv/bin/python" -c '
import sys, urllib.request
print(urllib.request.urlopen(
    "http://%s:%s%s" % (sys.argv[1], sys.argv[2], sys.argv[3]), timeout=3
).read().decode()[:400])' "$PROBE_HOST" "$PORT" "$1"
}

start_model_download() {
  "$APP_DIR/venv/bin/python" -c '
import json, sys, urllib.request
url = "http://%s:%s/api/pull" % (sys.argv[1], sys.argv[2])
payload = json.dumps({"model": sys.argv[3], "stream": False}).encode()
request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
print(urllib.request.urlopen(request, timeout=10).read().decode()[:400])' \
    "$PROBE_HOST" "$PORT" "$MODEL_ID"
}

printf '  waiting for the API on %s:%s ' "$PROBE_HOST" "$PORT"
API_UP=0
for _ in $(seq 1 30); do
  if probe /api/version >/dev/null 2>&1; then
    API_UP=1
    break
  fi
  printf '.'
  sleep 2
done
echo

if (( API_UP == 0 )); then
  echo "  the API did not answer within 60s"
  show_failure
  exit 1
fi

echo "  /api/version -> $(probe /api/version)"
MODEL_ID="$(read_env_value "$APP_DIR/.env" AIRLLM_MODEL_ID || true)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-72B-Instruct}"
echo "  initiating model download/sharding for $MODEL_ID"
echo "  /api/pull    -> $(start_model_download)"
echo "  /health      -> $(probe /health)"
echo
echo "--- systemctl status ${SERVICE_NAME} ---"
as_root systemctl status "${SERVICE_NAME}.service" --no-pager --lines=0 || true

cat <<DONE

=========================================================
${SERVICE_NAME}.service is running on port ${PORT}
(Ollama-compatible HTTP API served by AirLLM).

The model download/sharding has been initiated in the background.
/api/version and /api/tags answer now; /api/chat returns 503 until
the weights are ready — /health reports which.

  Check the service : systemctl status ${SERVICE_NAME}
  Follow the log    : ${JOURNAL} -f
  Load status       : curl -s ${PROBE_HOST}:${PORT}/health
  Smoke test        : curl -s ${PROBE_HOST}:${PORT}/api/chat -d '{
                        "model":"local","messages":[{"role":"user","content":"hi"}]
                      }'

Config lives in ${APP_DIR}/.env — edit it and run
'${SUDO_HINT}systemctl restart ${SERVICE_NAME}' to apply changes.
=========================================================
DONE
