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
DEFAULT_NETWORK_MODE="${DEFAULT_NETWORK_MODE:-static}"
DEFAULT_IP_ADDRESS="${DEFAULT_IP_ADDRESS:-}"
DEFAULT_SUBNET="${DEFAULT_SUBNET:-24}"
DEFAULT_GATEWAY="${DEFAULT_GATEWAY:-}"
DEFAULT_DNS="${DEFAULT_DNS:-}"
DEFAULT_OS_TYPE="${DEFAULT_OS_TYPE:-auto}"
DEFAULT_FEATURES="${DEFAULT_FEATURES:-nesting=1,keyctl=1}"
NETWORK_WAIT_SECONDS="${NETWORK_WAIT_SECONDS:-180}"
ALLOW_UNSTABLE_APT_RELEASE="${ALLOW_UNSTABLE_APT_RELEASE:-0}"
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

prompt_optional() {
  local label="$1"
  local default="$2"
  local value
  if [[ -n "$default" ]]; then
    printf '%s [%s]: ' "$label" "$default" >&2
  else
    printf '%s: ' "$label" >&2
  fi
  read -r value
  printf '%s' "${value:-$default}"
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

subnet_to_prefix() {
  local subnet="$1"
  local IFS=.
  local -a octets
  local octet bit prefix=0 seen_zero=0

  if [[ "$subnet" =~ ^[0-9]+$ ]]; then
    (( subnet >= 1 && subnet <= 32 )) || die "subnet prefix must be between 1 and 32, got: $subnet"
    printf '%s' "$subnet"
    return
  fi

  read -r -a octets <<< "$subnet"
  [[ ${#octets[@]} -eq 4 ]] || die "subnet mask must be CIDR bits or dotted quad, got: $subnet"
  for octet in "${octets[@]}"; do
    [[ "$octet" =~ ^[0-9]+$ ]] || die "subnet mask has a non-numeric octet: $subnet"
    (( octet >= 0 && octet <= 255 )) || die "subnet mask octet out of range: $subnet"
    for bit in 128 64 32 16 8 4 2 1; do
      if (( octet & bit )); then
        (( seen_zero == 0 )) || die "subnet mask must be contiguous, got: $subnet"
        prefix=$((prefix + 1))
      else
        seen_zero=1
      fi
    done
  done
  (( prefix >= 1 && prefix <= 32 )) || die "subnet prefix must be between 1 and 32, got: $prefix"
  printf '%s' "$prefix"
}

build_network_config() {
  local mode bridge ip_address subnet prefix gateway net0
  case "${DEFAULT_NETWORK_MODE,,}" in
    static) mode="$(prompt_choice "Network mode" "static" "dhcp")" ;;
    dhcp) mode="$(prompt_choice "Network mode" "dhcp" "static")" ;;
    *) die "DEFAULT_NETWORK_MODE must be static or dhcp, got: $DEFAULT_NETWORK_MODE" ;;
  esac
  bridge="$(prompt "Bridge" "$DEFAULT_BRIDGE")"

  case "${mode,,}" in
    dhcp)
      printf 'name=eth0,bridge=%s,ip=dhcp' "$bridge"
      ;;
    static)
      ip_address="$(prompt "IP address" "$DEFAULT_IP_ADDRESS")"
      subnet="$(prompt "Subnet mask or CIDR prefix" "$DEFAULT_SUBNET")"
      gateway="$(prompt "Default gateway" "$DEFAULT_GATEWAY")"
      prefix="$(subnet_to_prefix "$subnet")"
      net0="name=eth0,bridge=${bridge},ip=${ip_address}/${prefix},gw=${gateway}"
      printf '%s' "$net0"
      ;;
    *)
      die "unsupported network mode: $mode"
      ;;
  esac
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
  pvesm list --content vztmpl 2>/dev/null | awk 'NR > 1 {print $1}' | grep -Eq "(^|:)vztmpl/${template}$"
}

ensure_template_available() {
  local template="$1"
  if template_volume_exists "$template"; then
    note "template is already available: $template"
  else
    note "downloading template: $template"
    pveam download local "$template" >&2
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

warn_about_template() {
  local template="$1"
  case "$template" in
    *resolute*|*ubuntu-26.10*|*ubuntu-26-10*)
      printf '\nWARNING: %s appears to be an Ubuntu development/new-release template.\n' "$template" >&2
      printf 'Its apt repositories may not have normal Release files yet. For a reliable\n' >&2
      printf 'AirLLM install, use an Ubuntu LTS or Debian template.\n\n' >&2
      prompt_yes_no "Continue with this template anyway?" "no" || die "aborted"
      ;;
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

wait_for_container_network() {
  local vmid="$1"
  local timeout="${2:-$NETWORK_WAIT_SECONDS}"
  local elapsed=0
  printf '  waiting for outbound network in CT %s ' "$vmid"
  while (( elapsed < timeout )); do
    if pct exec "$vmid" -- sh -c 'ip route show default 2>/dev/null | grep -q . && getent hosts archive.ubuntu.com >/dev/null 2>&1' >/dev/null 2>&1; then
      printf '\n'
      return 0
    fi
    if pct exec "$vmid" -- sh -c 'ip route show default 2>/dev/null | grep -q . && getent hosts deb.debian.org >/dev/null 2>&1' >/dev/null 2>&1; then
      printf '\n'
      return 0
    fi
    printf '.'
    sleep 3
    elapsed=$((elapsed + 3))
  done
  printf '\n'
  die "container has no usable default route/DNS after ${timeout}s. Check the selected net0 config, bridge, DHCP, gateway, and DNS before running the AirLLM installer."
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
    ALLOW_UNSTABLE_APT_RELEASE="$ALLOW_UNSTABLE_APT_RELEASE" \
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
warn_about_template "$template"
ostype="$(ostype_for_template "$template")"
vmid="$(prompt "Container VMID" "$(next_vmid)")"
hostname="$(prompt "Hostname" "$DEFAULT_HOSTNAME")"
cores="$(prompt "CPU cores" "$DEFAULT_CORES")"
memory_mb="$(prompt "RAM in MB" "$DEFAULT_MEMORY_MB")"
swap_mb="$(prompt "Swap in MB" "$DEFAULT_SWAP_MB")"
storage="$(prompt_choice "Container-enabled storage targets" "${storages[@]}")"
disk_gb="$(prompt "Root disk size in GB" "$DEFAULT_DISK_GB")"
net0="$(build_network_config)"
nameserver="$(prompt_optional "DNS server (blank to skip)" "$DEFAULT_DNS")"
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
if (( ${#template_volume} > 255 )); then
  die "template volume is unexpectedly long; got: $template_volume"
fi

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
  DNS       : ${nameserver:-container default}
  OS type   : $ostype
  Unpriv    : $UNPRIVILEGED

SUMMARY

prompt_yes_no "Create this container?" "yes" || die "aborted"

create_args=(
  create "$vmid" "$template_volume"
  --hostname "$hostname"
  --cores "$cores"
  --memory "$memory_mb"
  --swap "$swap_mb"
  --rootfs "${storage}:${disk_gb}"
  --net0 "$net0"
  --ostype "$ostype"
  --features "$DEFAULT_FEATURES"
  --unprivileged "$UNPRIVILEGED"
  --password "$password"
  --start "$START_CONTAINER"
)
if [[ -n "$nameserver" ]]; then
  create_args+=(--nameserver "$nameserver")
fi
pct "${create_args[@]}"

if [[ "$START_CONTAINER" != "1" ]]; then
  note "starting CT $vmid"
  pct start "$vmid"
fi

wait_for_container_systemd "$vmid"
wait_for_container_network "$vmid"

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
