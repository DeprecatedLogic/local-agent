#!/usr/bin/env bash
set -euo pipefail

# local-agent sandbox bootstrap/launcher.
#
# Run this script as root to test:
#   sudo ./scripts/launch_agent_sandbox.sh /srv/local-agent-workspace
#
# The agent runs as the unprivileged "localagent" user. The host project is
# never used as its working tree; only the supplied workspace is.
#
# Override these before running if your launcher differs:
#   AGENT_MODULE=agent.cli
#   AGENT_VENV=/srv/local-agent-workspace/.venv
#   AGENT_UNIT=local-agent.service

AGENT_USER="${AGENT_USER:-localagent}"
AGENT_GROUP="${AGENT_GROUP:-localagent_sandbox}"
AGENT_HOME="${AGENT_HOME:-/var/lib/localagent}"
WORKSPACE="${1:-/srv/local-agent-workspace}"
AGENT_MODULE="${AGENT_MODULE:-agent.cli}"
AGENT_UNIT="${AGENT_UNIT:-local-agent.service}"
VENV="${AGENT_VENV:-${WORKSPACE}/.venv}"

SERVICE_PATH="/etc/systemd/system/${AGENT_UNIT}"
LAUNCHER_PATH="/usr/local/libexec/local-agent-launch"

die() {
    echo "error: $*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "run this script as root (for example: sudo $0 ...)"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_path() {
    local path="$1"
    [[ "$path" = /* ]] || die "workspace must be an absolute path: $path"
    [[ "$path" != "/" ]] || die "refusing to use / as workspace"
}

create_group() {
    if getent group "$AGENT_GROUP" >/dev/null; then
        echo "group exists: $AGENT_GROUP"
    else
        groupadd --system "$AGENT_GROUP"
        echo "created group: $AGENT_GROUP"
    fi
}

create_user() {
    if id "$AGENT_USER" >/dev/null 2>&1; then
        echo "user exists: $AGENT_USER"
    else
        useradd \
            --system \
            --home-dir "$AGENT_HOME" \
            --create-home \
            --shell /usr/sbin/nologin \
            "$AGENT_USER"
        echo "created user: $AGENT_USER"
    fi

    # Do not replace the user's existing supplementary groups.
    usermod --append --groups "$AGENT_GROUP" "$AGENT_USER"

    # The agent account must not have a password/login shell.
    passwd --lock "$AGENT_USER" >/dev/null 2>&1 || true
    usermod --shell /usr/sbin/nologin "$AGENT_USER"
}

prepare_home() {
    install -d -o "$AGENT_USER" -g "$AGENT_USER" -m 0700 "$AGENT_HOME"

    # The group is deliberately allowed to traverse/read the agent home, but
    # not modify it. This lets members inspect the environment without giving
    # them ownership of the agent's private state.
    setfacl -m "g:${AGENT_GROUP}:r-x" "$AGENT_HOME"
    setfacl -m "m::r-x" "$AGENT_HOME"
}

prepare_workspace() {
    install -d -o "$AGENT_USER" -g "$AGENT_GROUP" -m 2770 "$WORKSPACE"

    # Ensure the agent owns everything in its disposable workspace. This is
    # intentionally limited to the supplied workspace.
    chown -R "$AGENT_USER:$AGENT_GROUP" "$WORKSPACE"

    # New files/directories inherit the sandbox group.
    chmod 2770 "$WORKSPACE"

    # Allow group members (for example, DeprecatedLogic) to access the workspace.
    setfacl -m "g:${AGENT_GROUP}:rwx" "$WORKSPACE"
    setfacl -m "m::rwx" "$WORKSPACE"
    setfacl -d -m "g:${AGENT_GROUP}:rwx,m::rwx" "$WORKSPACE"
}

prepare_venv() {
    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "creating virtual environment: $VENV"
        install -d -o "$AGENT_USER" -g "$AGENT_GROUP" -m 2770 "$VENV"
        runuser -u "$AGENT_USER" -- python3 -m venv "$VENV"
    fi

    chown -R "$AGENT_USER:$AGENT_GROUP" "$VENV"
    chmod 2770 "$VENV"

    # Keep Python package installation inside this venv.
    runuser -u "$AGENT_USER" -- \
        "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
}

write_launcher() {
    install -d -m 0755 /usr/local/libexec

    cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export HOME=$(printf '%q' "$AGENT_HOME")
export VIRTUAL_ENV=$(printf '%q' "$VENV")
export PATH=$(printf '%q' "$VENV/bin"):/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1

cd $(printf '%q' "$WORKSPACE")

exec "$VENV/bin/python" -m "$AGENT_MODULE" --workspace "$WORKSPACE"
EOF

    chown root:root "$LAUNCHER_PATH"
    chmod 0755 "$LAUNCHER_PATH"
}

write_service() {
    local logical_cpus
    logical_cpus="$(getconf _NPROCESSORS_ONLN)"
    
    # Leave two logical CPUs outside the agent's CPU affinity when possible.
    # This is a hard CPU-placement boundary, not merely a scheduling hint.
    local allowed_cpus
    if (( logical_cpus > 2 )); then
        allowed_cpus="0-$((logical_cpus - 3))"
    else
        allowed_cpus="0"
    fi
    allowed_cpus="Unset"

    cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Local Agent sandbox
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${AGENT_USER}
Group=${AGENT_USER}
SupplementaryGroups=${AGENT_GROUP}

WorkingDirectory=${WORKSPACE}
Environment=HOME=${AGENT_HOME}
Environment=VIRTUAL_ENV=${VENV}
Environment=PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONNOUSERSITE=1
Environment=PYTHONDONTWRITEBYTECODE=1

ExecStart=${LAUNCHER_PATH}
Restart=no-failure
RestartSec=3

# Systemd logging
StandardOutput=journal
StandardError=journal

# Filesystem sandbox
ProtectSystem=strict
ProtectHome=tmpfs
ReadWritePaths=${AGENT_HOME}
ReadWritePaths=${WORKSPACE}
PrivateTmp=yes

# Hardware/kernel protection
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
LockPersonality=yes
NoNewPrivileges=yes
RestrictSUIDSGID=yes
CapabilityBoundingSet=

# Do not give the agent access to the system's service manager or user bus
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

# Resource controls:
CPUAccounting=yes
#AllowedCPUs=${allowed_cpus}
CPUWeight=10
Nice=10

# Hard ceiling: prevents runaway memory/process creation
MemoryMax=4G    # hard kill at 4G
MemoryHigh=3.5G   # soft throttle at 3.5G
TasksMax=256
LimitNPROC=256
LimitNOFILE=4096
LimitCORE=0

# Avoid accidental persistence through core dumps
UMask=0077

[Install]
WantedBy=multi-user.target
EOF

    chmod 0644 "$SERVICE_PATH"
    chown root:root "$SERVICE_PATH"

    echo "systemd unit written: $SERVICE_PATH"
    echo "agent CPU affinity: $allowed_cpus"
    echo "logical CPUs detected: $logical_cpus"
}

install_dependencies_if_present() {
    # Do this only if the project already declares dependencies. The agent
    # remains responsible for installing future dependencies inside the venv.
    if [[ -f "${WORKSPACE}/requirements.txt" ]]; then
        echo "installing requirements.txt into sandbox venv"
        runuser -u "$AGENT_USER" -- \
            "$VENV/bin/python" -m pip install -r "${WORKSPACE}/requirements.txt"
    elif [[ -f "${WORKSPACE}/pyproject.toml" ]]; then
        echo "installing project into sandbox venv"
        runuser -u "$AGENT_USER" -- \
            "$VENV/bin/python" -m pip install -e "$WORKSPACE"
    fi
}

verify() {
    echo
    echo "Sandbox configuration:"
    echo "  user:      $AGENT_USER"
    echo "  group:     $AGENT_GROUP"
    echo "  home:      $AGENT_HOME"
    echo "  workspace: $WORKSPACE"
    echo "  venv:      $VENV"
    echo "  unit:      $AGENT_UNIT"
    echo

    id "$AGENT_USER"
    stat -c 'home      %A %U:%G %n' "$AGENT_HOME"
    stat -c 'workspace %A %U:%G %n' "$WORKSPACE"
    "$VENV/bin/python" --version
}

main() {
    require_root

    require_cmd useradd
    require_cmd groupadd
    require_cmd usermod
    require_cmd runuser
    require_cmd setfacl
    require_cmd systemctl
    require_cmd python3
    require_cmd getconf

    validate_path "$WORKSPACE"

    create_group
    create_user
    prepare_home
    prepare_workspace
    prepare_venv
    install_dependencies_if_present
    write_launcher
    write_service

    systemctl daemon-reload
    systemctl enable "$AGENT_UNIT"

    verify

    echo
    echo "Sandbox installed."
    echo "Start with:"
    echo "  sudo systemctl start $AGENT_UNIT"
    echo
    echo "Inspect with:"
    echo "  systemctl status $AGENT_UNIT"
    echo "  journalctl -u $AGENT_UNIT -f"
    echo
    echo "The agent command is currently:"
    echo "  $VENV/bin/python -m $AGENT_MODULE"
    echo "If your CLI entry point differs, set AGENT_MODULE before running this script."
}

main "$@"
