#!/usr/bin/env bash
#
# Create a Proxmox LXC container for airllm-ollama-api, then run the in-guest
# Linux bootstrapper to install AirLLM and start the Ollama-compatible service.
#
# Run this on a Proxmox VE node as root:
#     ./proxmox-lxc-install.sh

set -Eeuo pipefail

DEFAULT_HOSTNAME="${DEFAULT_HOSTNAME:-airllm-ollama-api}"
DEFAULT_CORES="${DEFAULT_CORES:-4}"
DEFAULT_MEMORY_MB="${DEFAULT_MEMORY_MB:-32768}"
DEFAULT_SWAP_MB="${DEFAULT_SWAP_MB:-4096}"
DEFAULT_DISK_GB="${DEFAULT_DISK_GB:-300}"
DEFAULT_BRIDGE="${DEFAULT_BRIDGE:-vmbr0}"
DEFAULT_NET="${DEFAULT_NET:-name=eth0,bridge=${DEFAULT_BRIDGE},ip=dhcp}"
DEFAULT_OS_TYPE="${DEFAULT_OS_TYPE:-auto}"
DEFAULT_FEATURES="${DEFAULT_FEATURES:-nesting=1,keyctl=1}"
START_CONTAINER="${START_CONTAINER:-1}"
UNPRIVILEGED="${UNPRIVILEGED:-1}"
INSTALL_IN_CONTAINER="${INSTALL_IN_CONTAINER:-1}"
APP_DIR="${APP_DIR:-/opt/airllm-ollama-api}"
CONTAINER_SRC_DIR="${CONTAINER_SRC_DIR:-/root/airllm-ollama-api-src}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*" >&2; }

prompt() {
  local label="$1"
  local default="$2"
  local value
  if [[ -n "$default" ]]; then
    printf '%s [%s]: ' "$label" "$default" >&2
    read -r value
    printf '%s' "${value:-$default}"
  else
    printf '%s: ' "$label" >&2
    read -r value
    [[ -n "$value" ]] || die "$label is required"
    printf '%s' "$value"
  fi
}

prompt_secret() {
  local label="$1"
  local value
  printf '%s: ' "$label" >&2
  read -r -s value
  printf '\n' >&2
  [[ -n "$value" ]] || die "$label is required"
  printf '%s' "$value"
}

prompt_yes_no() {
  local label="$1"
  local default="$2"
  local value
  printf '%s [%s]: ' "$label" "$default" >&2
  read -r value
  value="${value:-$default}"
  case "${value,,}" in
    y|yes|true|1) return 0 ;;
    n|no|false|0) return 1 ;;
    *) die "expected yes or no, got: $value" ;;
  esac
}

prompt_choice() {
  local label="$1"
  shift
  local choices=("$@")
  local i selected
  [[ ${#choices[@]} -gt 0 ]] || die "no choices available for $label"

  printf '\n%s\n' "$label" >&2
  for i in "${!choices[@]}"; do
    printf '  %2d) %s\n' "$((i + 1))" "${choices[$i]}" >&2
  done

  while true; do
    printf 'Select 1-%s: ' "${#choices[@]}" >&2
    read -r selected
    if [[ "$selected" =~ ^[0-9]+$ ]] && (( selected >= 1 && selected <= ${#choices[@]} )); then
      printf '%s' "${choices[$((selected - 1))]}"
      return
    fi
    printf 'Invalid selection.\n' >&2
  done
}

validate_int() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a positive integer, got: $value"
  (( value > 0 )) || die "$label must be greater than zero"
}

require_root() {
  [[ $EUID -eq 0 ]] || die "run this script as root on a Proxmox VE node"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

list_templates() {
  pveam available --section system | awk 'NR > 1 {print $2}' | sort -u
}

list_container_storage() {
  pvesm status -content rootdir 2>/dev/null | awk 'NR > 1 && $1 != "" {print $1}' | sort -u
}

template_volume_exists() {
  local template="$1"
  [[ -f "/var/lib/vz/template/cache/$template" ]] && return 0
  pvesm list local --content vztmpl 2>/dev/null | awk 'NR > 1 {print $1}' | grep -Fxq "local:vztmpl/$template"
}

ensure_template_available() {
  local template="$1"
  if template_volume_exists "$template"; then
    note "template is already available: $template"
  else
    note "downloading template: $template"
    pveam download local "$template"
  fi
  printf 'local:vztmpl/%s' "$template"
}

next_vmid() {
  pvesh get /cluster/nextid
}

ostype_for_template() {
  local template="$1"
  if [[ "$DEFAULT_OS_TYPE" != "auto" ]]; then
    printf '%s' "$DEFAULT_OS_TYPE"
    return
  fi

  case "$template" in
    *ubuntu*) printf 'ubuntu' ;;
    *debian*) printf 'debian' ;;
    *alpine*) printf 'alpine' ;;
    *archlinux*) printf 'archlinux' ;;
    *centos*) printf 'centos' ;;
    *fedora*) printf 'fedora' ;;
    *gentoo*) printf 'gentoo' ;;
    *opensuse*) printf 'opensuse' ;;
    *) printf 'unmanaged' ;;
  esac
}

wait_for_container_systemd() {
  local vmid="$1"
  local timeout="${2:-180}"
  local elapsed=0
  printf '  waiting for systemd in CT %s ' "$vmid"
  while (( elapsed < timeout )); do
    if pct exec "$vmid" -- systemctl is-system-running >/dev/null 2>&1; then
      printf '\n'
      return 0
    fi
    if pct exec "$vmid" -- test -d /run/systemd/system >/dev/null 2>&1; then
      printf '\n'
      return 0
    fi
    printf '.'
    sleep 3
    elapsed=$((elapsed + 3))
  done
  printf '\n'
  die "container systemd did not become ready within ${timeout}s"
}

copy_repo_into_container() {
  local vmid="$1"
  note "copying repo into CT $vmid:$CONTAINER_SRC_DIR"
  pct exec "$vmid" -- mkdir -p "$CONTAINER_SRC_DIR"
  tar -C "$REPO_DIR" \
    --exclude .git \
    --exclude .venv \
    --exclude __pycache__ \
    --exclude .pytest_cache \
    -cf - . | pct exec "$vmid" -- tar -C "$CONTAINER_SRC_DIR" -xf -
  pct exec "$vmid" -- chmod +x "$CONTAINER_SRC_DIR/airllm-system-install.sh" "$CONTAINER_SRC_DIR/airllm-ollama-api-install.sh"
}

run_airllm_install() {
  local vmid="$1"
  note "running airllm-system-install.sh inside CT $vmid"
  pct exec "$vmid" -- env \
    APP_DIR="$APP_DIR" \
    SERVICE_USER=airllm \
    SERVICE_GROUP=airllm \
    "$CONTAINER_SRC_DIR/airllm-system-install.sh"
}

require_root
require_command pct
require_command pvesm
require_command pveam
require_command pvesh
require_command awk
require_command sort
require_command tar

mapfile -t templates < <(list_templates)
mapfile -t storages < <(list_container_storage)

[[ ${#templates[@]} -gt 0 ]] || die "no LXC templates were returned by pveam"
[[ ${#storages[@]} -gt 0 ]] || die "no Proxmox storage with rootdir content is enabled"

template="$(prompt_choice "Available LXC templates" "${templates[@]}")"
ostype="$(ostype_for_template "$template")"
vmid="$(prompt "Container VMID" "$(next_vmid)")"
hostname="$(prompt "Hostname" "$DEFAULT_HOSTNAME")"
cores="$(prompt "CPU cores" "$DEFAULT_CORES")"
memory_mb="$(prompt "RAM in MB" "$DEFAULT_MEMORY_MB")"
swap_mb="$(prompt "Swap in MB" "$DEFAULT_SWAP_MB")"
storage="$(prompt_choice "Container-enabled storage targets" "${storages[@]}")"
disk_gb="$(prompt "Root disk size in GB" "$DEFAULT_DISK_GB")"
net0="$(prompt "Network config" "$DEFAULT_NET")"
password="$(prompt_secret "Root password for the container")"

validate_int "Container VMID" "$vmid"
validate_int "CPU cores" "$cores"
validate_int "RAM in MB" "$memory_mb"
validate_int "Swap in MB" "$swap_mb"
validate_int "Root disk size in GB" "$disk_gb"

if pct status "$vmid" >/dev/null 2>&1; then
  die "container VMID already exists: $vmid"
fi

template_volume="$(ensure_template_available "$template")"

cat <<SUMMARY

About to create:
  VMID      : $vmid
  Hostname  : $hostname
  Template  : $template_volume
  CPU       : $cores cores
  RAM       : ${memory_mb}MB
  Swap      : ${swap_mb}MB
  Storage   : ${storage}:${disk_gb}GB
  Network   : $net0
  OS type   : $ostype
  Unpriv    : $UNPRIVILEGED

SUMMARY

prompt_yes_no "Create this container?" "yes" || die "aborted"

pct create "$vmid" "$template_volume" \
  --hostname "$hostname" \
  --cores "$cores" \
  --memory "$memory_mb" \
  --swap "$swap_mb" \
  --rootfs "${storage}:${disk_gb}" \
  --net0 "$net0" \
  --ostype "$ostype" \
  --features "$DEFAULT_FEATURES" \
  --unprivileged "$UNPRIVILEGED" \
  --password "$password" \
  --start "$START_CONTAINER"

if [[ "$START_CONTAINER" != "1" ]]; then
  note "starting CT $vmid"
  pct start "$vmid"
fi

wait_for_container_systemd "$vmid"

if [[ "$INSTALL_IN_CONTAINER" == "1" ]]; then
  copy_repo_into_container "$vmid"
  run_airllm_install "$vmid"
else
  note "skipping in-container install because INSTALL_IN_CONTAINER=$INSTALL_IN_CONTAINER"
fi

cat <<DONE

=========================================================
Container $vmid is ready.

Enter it with:
  pct enter $vmid

Follow AirLLM setup/model download progress:
  pct exec $vmid -- journalctl -u airllm-ollama-api -f

Check API health:
  pct exec $vmid -- curl -s 127.0.0.1:11434/health
=========================================================
DONE
