#!/bin/sh

# Shared Webots process control for the scenario launchers.

ensure_physwarm_controller_port_available() {
    physwarm_controller_port=${WEBOTS_CONTROLLER_URL#ipc://}
    physwarm_controller_port=${physwarm_controller_port%%/*}

    case "$physwarm_controller_port" in
        ''|*[!0-9]*)
            return 0
            ;;
    esac

    if command -v lsof >/dev/null 2>&1 && \
        lsof -nP -iTCP:"$physwarm_controller_port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Webots controller port $physwarm_controller_port is already in use." >&2
        echo "Close the existing Webots instance, or run this launcher with 'existing'." >&2
        return 1
    fi
}

start_physwarm_webots() {
    physwarm_mode=$1
    physwarm_repo_root=$2
    physwarm_webots_pid=""

    case "$physwarm_mode" in
        existing)
            return 0
            ;;
        fast)
            ensure_physwarm_controller_port_available || return 1
            physwarm_webots_bin="$WEBOTS_HOME/Contents/MacOS/webots"
            physwarm_world="$physwarm_repo_root/worlds/generated_world.wbt"
            if [ ! -x "$physwarm_webots_bin" ]; then
                echo "Webots executable not found: $physwarm_webots_bin" >&2
                return 1
            fi
            physwarm_webots_log=$(mktemp -t physwarm-webots.XXXXXX)
            "$physwarm_webots_bin" --batch --mode=fast --no-rendering --stdout --stderr --extern-urls "$physwarm_world" >"$physwarm_webots_log" 2>&1 &
            physwarm_webots_pid=$!
            echo "Webots fast mode started in background (log: $physwarm_webots_log)"
            ;;
        slow)
            ensure_physwarm_controller_port_available || return 1
            physwarm_webots_bin="$WEBOTS_HOME/Contents/MacOS/webots"
            physwarm_world="$physwarm_repo_root/worlds/generated_world.wbt"
            if [ ! -x "$physwarm_webots_bin" ]; then
                echo "Webots executable not found: $physwarm_webots_bin" >&2
                return 1
            fi
            "$physwarm_webots_bin" --mode=realtime "$physwarm_world" &
            physwarm_webots_pid=$!
            echo "Webots realtime GUI started."
            ;;
        *)
            echo "Usage: $0 [existing|slow|fast] [extra train_prey.py options]" >&2
            return 64
            ;;
    esac
}

stop_physwarm_webots() {
    if [ -n "${physwarm_webots_pid:-}" ]; then
        kill "$physwarm_webots_pid" 2>/dev/null || true
        wait "$physwarm_webots_pid" 2>/dev/null || true
    fi
}
