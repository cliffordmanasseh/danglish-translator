#!/usr/bin/env bash
# ==============================================================================
# Local Church Wi-Fi Audio Broadcast System - Production Turnkey Launcher
# ==============================================================================
# 1. Uses pre-compiled ./church-audio-server (and ./vendor for 100% offline build).
# 2. Configures offline NetworkManager Wi-Fi hotspot on 10.42.0.1/24 (Ch 6) with
#    as an OPEN network (no password) so anyone in church can join and listen.
# 3. Configures dnsmasq wildcard DNS (address=/#/10.42.0.1) + Port 80 captive
#    probe responses so iOS & Android stay locked onto the Wi-Fi without internet.
# 4. Opens DHCP/DNS/HTTP(80 & 8080)/HTTPS(8443)/WebRTC(8555) ports in UFW.
# 5. Preserves PipeWire/PulseAudio session environment even if run via sudo.
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REAL_USER="${SUDO_USER:-${USER:-$(id -un)}}"
REAL_UID="$(id -u "${REAL_USER}")"
REAL_HOME="$(getent passwd "${REAL_USER}" | cut -d: -f6)"

export PATH="${REAL_HOME}/.local/go/bin:/home/clmm/.local/go/bin:/usr/local/go/bin:$PATH:/usr/sbin:/sbin"

HOTSPOT_CON_NAME="ChurchAudioHotspot"
SSID="${SSID:-Church_Live_Audio}"
SERVER_IP="${SERVER_IP:-10.42.0.1}"
HTTP_PORT="${HTTP_PORT:-8080}"
HTTPS_PORT="${HTTPS_PORT:-8443}"
WEBRTC_PORT="${WEBRTC_PORT:-8555}"
AUDIO_BACKEND="${AUDIO_BACKEND:-pulse}"
AUDIO_INPUT="${AUDIO_INPUT:-default}"
DNSMASQ_CONF="/etc/NetworkManager/dnsmasq-shared.d/offline-audio.conf"
BINARY_NAME="${SCRIPT_DIR}/church-audio-server"
LOCAL_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --local|--no-hotspot)
      LOCAL_ONLY=1
      SERVER_IP="127.0.0.1"
      ;;
  esac
done

# Preserve desktop user's PipeWire / PulseAudio runtime socket even if run with sudo
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${REAL_UID}}"
if [[ -S "/run/user/${REAL_UID}/pulse/native" ]]; then
  export PULSE_SERVER="unix:/run/user/${REAL_UID}/pulse/native"
fi

SUDO=""
if [[ "${EUID}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

echo "======================================================================"
echo "  Holstebro Pinsekirke - Local Wi-Fi Audio Broadcast System"
echo "======================================================================"

# 1. Verify required tools
for cmd in nmcli ffmpeg; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] Required command '${cmd}' is not installed." >&2
    exit 1
  fi
done

# 2. Verify pre-built binary exists (or build offline from ./vendor BEFORE hotspot starts)
if [[ ! -x "${BINARY_NAME}" ]] || [[ "main.go" -nt "${BINARY_NAME}" ]] || [[ "index.html" -nt "${BINARY_NAME}" ]] || [[ "speaker.html" -nt "${BINARY_NAME}" ]]; then
  if command -v go >/dev/null 2>&1; then
    echo "[INFO] Building ${BINARY_NAME} from local vendor cache..."
    CGO_ENABLED=0 go build -mod=vendor -ldflags="-s -w" -o "${BINARY_NAME}" main.go
  elif [[ ! -x "${BINARY_NAME}" ]]; then
    echo "[ERROR] Binary ${BINARY_NAME} not found." >&2
    exit 1
  fi
fi

WIFI_IFACE=""
PREV_WIFI_CON=""
CLEANUP_DONE=0

cleanup() {
  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1
  echo ""
  echo "[INFO] Shutting down broadcast server..."
  if [[ "${LOCAL_ONLY}" -eq 0 && -n "${WIFI_IFACE}" ]]; then
    echo "[INFO] Tearing down hotspot '${HOTSPOT_CON_NAME}' and restoring previous Wi-Fi..."
    nmcli connection down "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || ${SUDO} nmcli connection down "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || true
    nmcli connection delete "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || ${SUDO} nmcli connection delete "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || true
    if [[ -n "${PREV_WIFI_CON}" && "${PREV_WIFI_CON}" != "--" && "${PREV_WIFI_CON}" != "${HOTSPOT_CON_NAME}" ]]; then
      nmcli connection up "${PREV_WIFI_CON}" >/dev/null 2>&1 || true
    fi
  fi
  echo "[INFO] Restored normal Wi-Fi configuration. Goodbye."
}
trap cleanup EXIT INT TERM

if [[ "${LOCAL_ONLY}" -eq 0 ]]; then
  WIFI_IFACE=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')
  if [[ -z "${WIFI_IFACE}" ]]; then
    echo "[ERROR] No Wi-Fi interface found via NetworkManager." >&2
    exit 1
  fi
  PREV_WIFI_CON=$(nmcli -t -f DEVICE,CONNECTION device status | awk -F: -v dev="${WIFI_IFACE}" '$1==dev{print $2; exit}')
  echo "[INFO] Wi-Fi Interface : ${WIFI_IFACE} (previous connection: ${PREV_WIFI_CON:-none})"

  echo "[INFO] Applying hotspot network, DNS, and firewall settings..."
  if ${SUDO} true 2>/dev/null; then
    ${SUDO} mkdir -p /etc/NetworkManager/dnsmasq-shared.d
    # Provide normal default gateway (10.42.0.1) so older iPhones route traffic over Wi-Fi,
    # and resolve all domains (captive.apple.com, connectivitycheck.gstatic.com, audio.lan, etc.)
    # to 10.42.0.1 so our Port 80 probe responder keeps iOS & Android locked onto Wi-Fi.
    ${SUDO} tee "${DNSMASQ_CONF}" >/dev/null <<EOF
address=/#/${SERVER_IP}
EOF

    # Allow unprivileged binding to Port 80 & DSCP QoS marking
    ${SUDO} sysctl -w net.ipv4.ip_unprivileged_port_start=80 >/dev/null 2>&1 || true
    if command -v setcap >/dev/null 2>&1; then
      ${SUDO} setcap 'cap_net_bind_service,cap_net_admin,cap_net_raw+ep' "${BINARY_NAME}" >/dev/null 2>&1 || true
    fi

    if command -v ufw >/dev/null 2>&1 && ${SUDO} ufw status | grep -q "Status: active"; then
      echo "[INFO] Allowing Hotspot DHCP, DNS, HTTP (80 & ${HTTP_PORT}), HTTPS (${HTTPS_PORT}), and WebRTC (${WEBRTC_PORT}/udp) in UFW..."
      ${SUDO} ufw allow in on "${WIFI_IFACE}" to any port 67 proto udp >/dev/null 2>&1 || true
      ${SUDO} ufw allow in on "${WIFI_IFACE}" to any port 53 >/dev/null 2>&1 || true
      ${SUDO} ufw allow 80/tcp >/dev/null 2>&1 || true
      ${SUDO} ufw allow "${HTTP_PORT}/tcp" >/dev/null 2>&1 || true
      ${SUDO} ufw allow "${HTTPS_PORT}/tcp" >/dev/null 2>&1 || true
      ${SUDO} ufw allow "${WEBRTC_PORT}/udp" >/dev/null 2>&1 || true
    fi

    if command -v firewall-cmd >/dev/null 2>&1 && systemctl is-active --quiet firewalld; then
      ${SUDO} firewall-cmd --add-port="80/tcp" >/dev/null 2>&1 || true
      ${SUDO} firewall-cmd --add-port="${HTTP_PORT}/tcp" >/dev/null 2>&1 || true
      ${SUDO} firewall-cmd --add-port="${HTTPS_PORT}/tcp" >/dev/null 2>&1 || true
      ${SUDO} firewall-cmd --add-port="${WEBRTC_PORT}/udp" >/dev/null 2>&1 || true
    fi
  fi

  echo "[INFO] Starting Wi-Fi Hotspot '${SSID}' on ${WIFI_IFACE} (${SERVER_IP}/24, 2.4GHz Channel 6)..."
  nmcli connection down "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || ${SUDO} nmcli connection down "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || true
  nmcli connection delete "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || ${SUDO} nmcli connection delete "${HOTSPOT_CON_NAME}" >/dev/null 2>&1 || true

  nmcli connection add type wifi ifname "${WIFI_IFACE}" con-name "${HOTSPOT_CON_NAME}" autoconnect no ssid "${SSID}" >/dev/null
  nmcli connection modify "${HOTSPOT_CON_NAME}" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    802-11-wireless.channel 6 \
    ipv4.method shared \
    ipv4.addresses "${SERVER_IP}/24" \
    ipv6.method ignore
  # No wifi-sec settings: the hotspot is an open network (no password), on purpose.
  # The connection is deleted and re-created above, so no old password survives.

  nmcli connection up "${HOTSPOT_CON_NAME}" >/dev/null

  if command -v iw >/dev/null 2>&1 && ${SUDO} -n true 2>/dev/null; then
    ${SUDO} iw dev "${WIFI_IFACE}" set power_save off >/dev/null 2>&1 || true
  fi
  echo "[INFO] Wi-Fi Hotspot '${SSID}' is LIVE at ${SERVER_IP}!"
fi

# Run the broadcast server as the desktop user so PipeWire/PulseAudio works natively
if [[ "${EUID}" -eq 0 && "${REAL_USER}" != "root" ]]; then
  sudo -u "${REAL_USER}" env \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
    PULSE_SERVER="${PULSE_SERVER:-}" \
    "${BINARY_NAME}" \
      --ip="${SERVER_IP}" \
      --http-port="${HTTP_PORT}" \
      --https-port="${HTTPS_PORT}" \
      --webrtc-port="${WEBRTC_PORT}" \
      --audio-backend="${AUDIO_BACKEND}" \
      --audio-input="${AUDIO_INPUT}" \
      --ssid="${SSID}"
else
  "${BINARY_NAME}" \
    --ip="${SERVER_IP}" \
    --http-port="${HTTP_PORT}" \
    --https-port="${HTTPS_PORT}" \
    --webrtc-port="${WEBRTC_PORT}" \
    --audio-backend="${AUDIO_BACKEND}" \
    --audio-input="${AUDIO_INPUT}" \
    --ssid="${SSID}"
fi
