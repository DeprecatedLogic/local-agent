#!/usr/bin/env bash
set -euo pipefail

# Install and run local-agent inside a hardened, interactive systemd sandbox.
#
# Installation:
#   sudo ./scripts/local-agent-sandbox.sh
#
# Optional workspace override:
#   sudo ./scripts/local-agent-sandbox.sh /srv/my-agent-workspace
#
# Runtime:
#   sudo local-agent-sandbox
#   sudo local-agent-sandbox --chat
#   sudo local-agent-sandbox "Inspect the workspace and summarize it."
#
# The agent runtime is installed read-only under /opt/local-agent.
# The writable project workspace is kept separate under /srv.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_SOURCE="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

AGENT_USER="${AGENT_USER:-localagent}"
AGENT_GROUP="${AGENT_GROUP:-localagent_sandbox}"
AGENT_HOME="${AGENT_HOME:-/var/lib/localagent}"

AGENT_SOURCE="${AGENT_SOURCE:-$DEFAULT_SOURCE}"
WORKSPACE="${1:-${LOCAL_AGENT_WORKSPACE:-/srv/local-agent-workspace}}"

INSTALL_ROOT="${AGENT_INSTALL_ROOT:-/opt/local-agent}"
VENV="${AGENT_VENV:-${INSTALL_ROOT}/venv}"
LAUNCHER_PATH="${AGENT_LAUNCHER:-/usr/local/bin/local-agent-sandbox}"

# The old implementation used a persistent boot service. Chat mode needs a PTY,
# so the new launcher uses transient systemd services instead.
LEGACY_UNIT="${AGENT_UNIT:-local-agent.service}"
LEGACY_SERVICE_PATH="/etc/systemd/system/${LEGACY_UNIT}"
LEGACY_LAUNCHER_PATH="/usr/local/libexec/local-agent-launch"

# Give the user who invoked sudo access to the shared workspace as well.
ACCESS_USER="${SANDBOX_ACCESS_USER:-${SUDO_USER:-}}"

MEMORY_MAX="${AGENT_MEMORY_MAX:-4G}"
MEMORY_HIGH="${AGENT_MEMORY_HIGH:-3584M}"
TASKS_MAX="${AGENT_TASKS_MAX:-256}"

die() {
    echo "error: $*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || \
        die "run this installer as root (for example: sudo $0)"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || \
        die "required command not found: $1"
}

normalize_and_validate_paths() {
    AGENT_SOURCE="$(realpath "$AGENT_SOURCE")"
    WORKSPACE="$(realpath -m "$WORKSPACE")"
    INSTALL_ROOT="$(realpath -m "$INSTALL_ROOT")"
    VENV="$(realpath -m "$VENV")"
    LAUNCHER_PATH="$(realpath -m "$LAUNCHER_PATH")"

    [[ -d "$AGENT_SOURCE" ]] || \
        die "agent source directory does not exist: $AGENT_SOURCE"

    [[ -f "${AGENT_SOURCE}/pyproject.toml" ]] || \
        die "pyproject.toml not found in agent source: $AGENT_SOURCE"

    [[ "$WORKSPACE" != "/" ]] || \
        die "refusing to use / as workspace"

    [[ "$WORKSPACE" != "$AGENT_SOURCE" ]] || \
        die "agent source and sandbox workspace must be different directories"

    case "$WORKSPACE" in
        "$AGENT_SOURCE"/*)
            die "sandbox workspace must not be inside the agent source tree"
            ;;
    esac

    # Protect against accidentally chowning a normal home directory/project.
    case "$WORKSPACE" in
        /srv/*|/var/lib/localagent/workspaces/*)
            ;;
        *)
            die \
                "sandbox workspace must live under /srv or " \
                "/var/lib/localagent/workspaces: $WORKSPACE"
            ;;
    esac
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

    usermod \
        --home "$AGENT_HOME" \
        --shell /usr/sbin/nologin \
        --append \
        --groups "$AGENT_GROUP" \
        "$AGENT_USER"

    passwd --lock "$AGENT_USER" >/dev/null 2>&1 || true
}

configure_access_user() {
    if [[ -z "$ACCESS_USER" || "$ACCESS_USER" == "root" ]]; then
        return
    fi

    if ! id "$ACCESS_USER" >/dev/null 2>&1; then
        echo "warning: access user does not exist: $ACCESS_USER" >&2
        return
    fi

    usermod --append --groups "$AGENT_GROUP" "$ACCESS_USER"

    echo "workspace access user: $ACCESS_USER"
}

prepare_home() {
    install -d \
        -o "$AGENT_USER" \
        -g "$AGENT_USER" \
        -m 0700 \
        "$AGENT_HOME"
}

prepare_workspace() {
    install -d \
        -o "$AGENT_USER" \
        -g "$AGENT_GROUP" \
        -m 2770 \
        "$WORKSPACE"

    # This path is intentionally dedicated to the sandbox.
    chown -R "$AGENT_USER:$AGENT_GROUP" "$WORKSPACE"
    chmod 2770 "$WORKSPACE"

    # Existing content gets group rwX; new content inherits the sandbox group.
    setfacl -R -m "g:${AGENT_GROUP}:rwX" "$WORKSPACE"
    setfacl -m "d:g:${AGENT_GROUP}:rwx,d:m::rwx" "$WORKSPACE"

    if [[ -n "$ACCESS_USER" && "$ACCESS_USER" != "root" ]] \
        && id "$ACCESS_USER" >/dev/null 2>&1; then

        setfacl -R -m "u:${ACCESS_USER}:rwX" "$WORKSPACE"
        setfacl -m "d:u:${ACCESS_USER}:rwx,d:m::rwx" "$WORKSPACE"
    fi
}

install_runtime() {
    echo "installing local-agent runtime from: $AGENT_SOURCE"

    install -d \
        -o root \
        -g root \
        -m 0755 \
        "$INSTALL_ROOT"

    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "creating runtime virtual environment: $VENV"
        python3 -m venv "$VENV"
    fi

    "${VENV}/bin/python" \
        -m pip install \
        --upgrade \
        pip setuptools wheel

    # Ensure dependencies are installed.
    "${VENV}/bin/python" \
        -m pip install \
        --upgrade \
        "$AGENT_SOURCE"

    # Always refresh local-agent itself even if pyproject.toml still has the
    # same version number.
    "${VENV}/bin/python" \
        -m pip install \
        --force-reinstall \
        --no-deps \
        "$AGENT_SOURCE"

    # The sandboxed agent can execute its runtime but cannot modify itself.
    chown -R root:root "$INSTALL_ROOT"
    chmod -R go-w "$INSTALL_ROOT"
}

remove_legacy_service() {
    systemctl disable --now "$LEGACY_UNIT" >/dev/null 2>&1 || true

    if [[ -e "$LEGACY_SERVICE_PATH" ]]; then
        rm -f "$LEGACY_SERVICE_PATH"
        echo "removed legacy service: $LEGACY_SERVICE_PATH"
    fi

    if [[ -e "$LEGACY_LAUNCHER_PATH" ]]; then
        rm -f "$LEGACY_LAUNCHER_PATH"
        echo "removed legacy launcher: $LEGACY_LAUNCHER_PATH"
    fi

    systemctl daemon-reload
    systemctl reset-failed "$LEGACY_UNIT" >/dev/null 2>&1 || true
}

write_launcher() {
    install -d \
        -o root \
        -g root \
        -m 0755 \
        "$(dirname "$LAUNCHER_PATH")"

    cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

AGENT_USER=$(printf '%q' "$AGENT_USER")
AGENT_GROUP=$(printf '%q' "$AGENT_GROUP")
AGENT_HOME=$(printf '%q' "$AGENT_HOME")
WORKSPACE=$(printf '%q' "$WORKSPACE")
VENV=$(printf '%q' "$VENV")
MEMORY_MAX=$(printf '%q' "$MEMORY_MAX")
MEMORY_HIGH=$(printf '%q' "$MEMORY_HIGH")
TASKS_MAX=$(printf '%q' "$TASKS_MAX")

if (( EUID != 0 )); then
    echo "error: run with sudo: sudo local-agent-sandbox [arguments...]" >&2
    exit 1
fi

# The workspace is fixed by the sandbox installer. Do not allow an invocation
# to point the agent at another part of the host filesystem.
for arg in "\$@"; do
    case "\$arg" in
        --workspace|--workspace=*)
            echo "error: --workspace is fixed by the sandbox installer: \$WORKSPACE" >&2
            exit 2
            ;;
    esac
done

# Running the sandbox without arguments means interactive chat.
if (( \$# == 0 )); then
    set -- --chat
fi

SYSTEMD_ARGS=(
    --quiet
    --wait
    --collect
    --pty
    --service-type=exec

    --property="User=\$AGENT_USER"
    --property="Group=\$AGENT_USER"
    --property="SupplementaryGroups=\$AGENT_GROUP"
    --property="WorkingDirectory=\$WORKSPACE"

    --property="ProtectSystem=strict"
    --property="ProtectHome=tmpfs"
    --property="ReadWritePaths=\$AGENT_HOME"
    --property="ReadWritePaths=\$WORKSPACE"
    --property="PrivateTmp=yes"

    --property="PrivateDevices=yes"
    --property="ProtectKernelTunables=yes"
    --property="ProtectKernelModules=yes"
    --property="ProtectKernelLogs=yes"
    --property="ProtectControlGroups=yes"
    --property="ProtectClock=yes"
    --property="ProtectHostname=yes"
    --property="LockPersonality=yes"
    --property="NoNewPrivileges=yes"
    --property="RestrictSUIDSGID=yes"
    --property="CapabilityBoundingSet="
    --property="RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6"

    --property="CPUWeight=10"
    --property="Nice=10"
    --property="MemoryMax=\$MEMORY_MAX"
    --property="MemoryHigh=\$MEMORY_HIGH"
    --property="TasksMax=\$TASKS_MAX"
    --property="LimitNPROC=\$TASKS_MAX"
    --property="LimitNOFILE=4096"
    --property="LimitCORE=0"

    # Files remain available to the sandbox group but inaccessible to others.
    --property="UMask=0007"

    --setenv="HOME=\$AGENT_HOME"
    --setenv="VIRTUAL_ENV=\$VENV"
    --setenv="PATH=\$VENV/bin:/usr/local/bin:/usr/bin:/bin"
    --setenv="PYTHONNOUSERSITE=1"
    --setenv="PYTHONDONTWRITEBYTECODE=1"
    --setenv="PYTHONUNBUFFERED=1"
)

if [[ -n "\${TERM:-}" ]]; then
    SYSTEMD_ARGS+=(--setenv="TERM=\$TERM")
fi

if [[ -n "\${COLORTERM:-}" ]]; then
    SYSTEMD_ARGS+=(--setenv="COLORTERM=\$COLORTERM")
fi

if [[ -n "\${LANG:-}" ]]; then
    SYSTEMD_ARGS+=(--setenv="LANG=\$LANG")
fi

exec systemd-run \
    "\${SYSTEMD_ARGS[@]}" \
    "\$VENV/bin/local-agent" \
    --workspace "\$WORKSPACE" \
    "\$@"
EOF

    chown root:root "$LAUNCHER_PATH"
    chmod 0755 "$LAUNCHER_PATH"
}

verify() {
    echo
    echo "Verifying installation..."

    [[ -x "${VENV}/bin/local-agent" ]] || \
        die "console entry point missing: ${VENV}/bin/local-agent"

    runuser \
        -u "$AGENT_USER" \
        -- \
        "${VENV}/bin/python" \
        -c 'import agent, agent.cli; print("agent import: ok")'

    "${VENV}/bin/local-agent" --help >/dev/null

    echo
    echo "Sandbox configuration:"
    echo "  source:      $AGENT_SOURCE"
    echo "  runtime:     $INSTALL_ROOT"
    echo "  venv:        $VENV"
    echo "  user:        $AGENT_USER"
    echo "  group:       $AGENT_GROUP"
    echo "  home:        $AGENT_HOME"
    echo "  workspace:   $WORKSPACE"
    echo "  launcher:    $LAUNCHER_PATH"

    if [[ -n "$ACCESS_USER" && "$ACCESS_USER" != "root" ]]; then
        echo "  access user: $ACCESS_USER"
    fi

    echo

    stat \
        -c 'runtime   %A %U:%G %n' \
        "$INSTALL_ROOT"

    stat \
        -c 'workspace %A %U:%G %n' \
        "$WORKSPACE"
}

main() {
    require_root

    require_cmd useradd
    require_cmd groupadd
    require_cmd usermod
    require_cmd passwd
    require_cmd runuser
    require_cmd setfacl
    require_cmd systemctl
    require_cmd systemd-run
    require_cmd python3
    require_cmd realpath
    require_cmd install
    require_cmd getent

    normalize_and_validate_paths

    create_group
    create_user
    configure_access_user

    prepare_home
    prepare_workspace

    install_runtime

    remove_legacy_service
    write_launcher

    verify

    echo
    echo "Sandbox installed."
    echo
    echo "Interactive chat:"
    echo "  sudo local-agent-sandbox"
    echo "  sudo local-agent-sandbox --chat"
    echo
    echo "One-shot task:"
    echo \
        '  sudo local-agent-sandbox "Inspect the workspace and summarize it."'
    echo
    echo "The old persistent local-agent.service is no longer used."
    echo \
        "Each invocation runs as a transient, hardened systemd service " \
        "attached to your terminal."
    echo
    echo \
        "After changing local-agent source code, rerun this installer " \
        "to refresh the isolated runtime."
}

main "$@"
