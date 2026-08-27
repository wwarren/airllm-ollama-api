#!/usr/bin/env bash
#
# Bootstrap a Linux host for airllm-ollama-api.
#
# This installs the system packages AirLLM commonly needs, creates a dedicated
# service account and cache directories, writes a starter .env when one does
# not already exist, then delegates the API/service install to
# ./airllm-ollama-api-install.sh.
#
# Safe to re-run. Existing .env files are left in place.

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/airllm-ollama-api}"
SERVICE_NAME="${SERVICE_NAME:-airllm-ollama-api}"
SERVICE_USER="${SERVICE_USER:-airllm}"
SERVICE_GROUP="${SERVICE_GROUP:-$SERVICE_USER}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_API_SERVICE="${INSTALL_API_SERVICE:-1}"
INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES:-0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_STEPS=6

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n[%s/%s] %s\n' "$1" "$TOTAL_STEPS" "$2"; }
note() { printf '  %s\n' "$*"; }

as_root() {
  if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

as_service_user() {
  if [[ $EUID -eq 0 ]]; then
    runuser -u "$SERVICE_USER" -- "$@"
  else
    sudo -u "$SERVICE_USER" "$@"
  fi
}

validate_name() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[A-Za-z0-9_.@-]+$ ]] || die "$label contains invalid characters: $value"
}

validate_app_dir() {
  [[ -n "$APP_DIR" && "$APP_DIR" = /* ]] || die "APP_DIR must be an absolute path"
  APP_DIR="$(readlink -m "$APP_DIR")"
  case "$APP_DIR" in
    /|/opt|/usr|/etc|/var|/home|/boot|/efi|/srv)
      die "refusing unsafe APP_DIR: $APP_DIR"
      ;;
    /usr/*|/etc/*|/boot/*|/efi/*)
      die "APP_DIR is read-only for the service under ProtectSystem=full: $APP_DIR"
      ;;
  esac
}

install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    as_root apt-get update
    as_root apt-get install -y \
      ca-certificates curl git build-essential pkg-config \
      "$PYTHON_BIN" python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y \
      ca-certificates curl git gcc gcc-c++ make pkgconf-pkg-config \
      "$PYTHON_BIN" python3-pip
  elif command -v yum >/dev/null 2>&1; then
    as_root yum install -y \
      ca-certificates curl git gcc gcc-c++ make pkgconfig \
      "$PYTHON_BIN" python3-pip
  elif command -v zypper >/dev/null 2>&1; then
    as_root zypper --non-interactive install \
      ca-certificates curl git gcc gcc-c++ make pkg-config \
      "$PYTHON_BIN" python3-pip python3-virtualenv
  elif command -v pacman >/dev/null 2>&1; then
    as_root pacman -Sy --needed --noconfirm \
      ca-certificates curl git base-devel python python-pip
  elif command -v apk >/dev/null 2>&1; then
    as_root apk add --no-cache \
      ca-certificates curl git build-base pkgconfig python3 py3-pip
  else
    die "no supported package manager found (apt, dnf, yum, zypper, pacman, apk)"
  fi
}

ensure_service_user() {
  if getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    note "group exists: $SERVICE_GROUP"
  else
    as_root groupadd --system "$SERVICE_GROUP"
    note "created group: $SERVICE_GROUP"
  fi

  if id -u "$SERVICE_USER" >/dev/null 2>&1; then
    note "user exists: $SERVICE_USER"
  else
    as_root useradd \
      --system \
      --gid "$SERVICE_GROUP" \
      --home-dir "$APP_DIR" \
      --shell /usr/sbin/nologin \
      "$SERVICE_USER"
    note "created user: $SERVICE_USER"
  fi
}

append_env_if_missing() {
  local key="$1"
  local value="$2"
  local file="$3"
  if ! grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
    printf '%s=%s\n' "$key" "$value" | as_root tee -a "$file" >/dev/null
  fi
}

trap 'die "failed on line $LINENO"' ERR

validate_name "SERVICE_NAME" "$SERVICE_NAME"
validate_name "SERVICE_USER" "$SERVICE_USER"
validate_name "SERVICE_GROUP" "$SERVICE_GROUP"
command -v readlink >/dev/null 2>&1 || die "readlink is required"
validate_app_dir
[[ -x "$REPO_DIR/airllm-ollama-api-install.sh" ]] || die "airllm-ollama-api-install.sh must exist and be executable"
[[ -f "$REPO_DIR/env.example" ]] || die "env.example not found next to this script"

if [[ $EUID -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root"
  sudo -v || die "sudo authentication failed"
fi

step 1 "Checking systemd"
if ! command -v systemctl >/dev/null 2>&1 || [[ ! -d /run/systemd/system ]]; then
  die "systemd is not running. On WSL, enable it with 'systemd=true' under [boot] in /etc/wsl.conf, then run 'wsl --shutdown'."
fi
note "systemd is available"

step 2 "Installing system packages"
install_packages

step 3 "Creating the AirLLM service account"
ensure_service_user

step 4 "Preparing AirLLM cache and shard directories"
as_root mkdir -p "$APP_DIR" "$APP_DIR/hf-cache" "$APP_DIR/shards"
as_root chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"
as_root chmod 0750 "$APP_DIR" "$APP_DIR/hf-cache" "$APP_DIR/shards"
note "APP_DIR=$APP_DIR"
note "HF_HOME=$APP_DIR/hf-cache"
note "AIRLLM_LAYER_SHARDS_PATH=$APP_DIR/shards"

step 5 "Preparing .env"
CREATED_ENV_FROM_EXAMPLE=0
if [[ -f "$APP_DIR/.env" ]]; then
  note "keeping existing $APP_DIR/.env"
elif [[ -f "$REPO_DIR/.env" ]]; then
  as_root install -m 0600 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$REPO_DIR/.env" "$APP_DIR/.env"
  note "copied .env from the repo"
else
  as_root install -m 0600 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$REPO_DIR/env.example" "$APP_DIR/.env"
  CREATED_ENV_FROM_EXAMPLE=1
  note "created $APP_DIR/.env from env.example"
fi
if [[ "$CREATED_ENV_FROM_EXAMPLE" == "1" ]]; then
  as_root sed -i -E "s|^[#[:space:]]*HF_HOME=.*|HF_HOME=${APP_DIR}/hf-cache|" "$APP_DIR/.env"
fi
append_env_if_missing "HF_HOME" "$APP_DIR/hf-cache" "$APP_DIR/.env"
append_env_if_missing "AIRLLM_LAYER_SHARDS_PATH" "$APP_DIR/shards" "$APP_DIR/.env"
append_env_if_missing "AIRLLM_DEVICE" "cpu" "$APP_DIR/.env"
as_root chown "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR/.env"
as_root chmod 0600 "$APP_DIR/.env"

if command -v nvidia-smi >/dev/null 2>&1; then
  note "nvidia-smi found. To use CUDA, set AIRLLM_DEVICE=cuda:0 in $APP_DIR/.env."
else
  note "no nvidia-smi found; leaving AIRLLM_DEVICE=cpu for RAM-based layer swapping"
fi

step 6 "Installing the airllm-ollama-api service"
if [[ "$INSTALL_API_SERVICE" == "1" ]]; then
  as_root env \
    APP_DIR="$APP_DIR" \
    SERVICE_NAME="$SERVICE_NAME" \
    SERVICE_USER="$SERVICE_USER" \
    SERVICE_GROUP="$SERVICE_GROUP" \
    PYTHON_BIN="$PYTHON_BIN" \
    "$REPO_DIR/airllm-ollama-api-install.sh"

  if [[ "$INSTALL_BITSANDBYTES" == "1" ]]; then
    note "installing bitsandbytes into $APP_DIR/venv"
    as_service_user "$APP_DIR/venv/bin/pip" install bitsandbytes
  fi
else
  note "skipping service install because INSTALL_API_SERVICE=$INSTALL_API_SERVICE"
fi

cat <<DONE

=========================================================
AirLLM host setup is complete.

Config file : $APP_DIR/.env
Service     : ${SERVICE_NAME}.service
API URL     : http://127.0.0.1:11434

For gated Hugging Face models, add HF_TOKEN to $APP_DIR/.env:
  sudo nano $APP_DIR/.env

Then restart after config changes:
  sudo systemctl restart ${SERVICE_NAME}

Watch model download/sharding progress:
  sudo journalctl -u ${SERVICE_NAME} -f
=========================================================
DONE
