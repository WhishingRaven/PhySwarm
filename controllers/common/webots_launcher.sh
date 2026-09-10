#!/bin/sh

# Shared Webots process control for the scenario launchers.

PHYSWARM_WEBOTS_STARTUP_TIMEOUT=${PHYSWARM_WEBOTS_STARTUP_TIMEOUT:-30}

physwarm_controller_port() {
    physwarm_url=${WEBOTS_CONTROLLER_URL:-}
    physwarm_port=${physwarm_url#ipc://}
    physwarm_port=${physwarm_port%%/*}

    case "$physwarm_port" in
        ''|*[!0-9]*)
            return 1
            ;;
    esac

    printf '%s\n' "$physwarm_port"
}

physwarm_controller_name() {
    physwarm_url=${WEBOTS_CONTROLLER_URL:-}
    physwarm_name=${physwarm_url#ipc://}
    physwarm_name=${physwarm_name#*/}
    physwarm_name=${physwarm_name%%/*}

    if [ -z "$physwarm_name" ] || [ "$physwarm_name" = "$physwarm_url" ]; then
        return 1
    fi

    printf '%s\n' "$physwarm_name"
}

physwarm_extern_endpoint() {
    physwarm_port=$(physwarm_controller_port) || return 1
    physwarm_name=$(physwarm_controller_name) || return 1

    printf '/tmp/webots/%s/%s/ipc/%s/extern\n' \
        "$USER" \
        "$physwarm_port" \
        "$physwarm_name"
}

ensure_physwarm_controller_url_valid() {
    if ! physwarm_controller_port >/dev/null 2>&1 || \
       ! physwarm_controller_name >/dev/null 2>&1; then
        echo "Invalid WEBOTS_CONTROLLER_URL: ${WEBOTS_CONTROLLER_URL:-<unset>}" >&2
        echo "Expected format: ipc://<port>/<controller-name>" >&2
        return 1
    fi
}

ensure_physwarm_controller_port_available() {
    ensure_physwarm_controller_url_valid || return 1

    physwarm_port=$(physwarm_controller_port) || return 1
    physwarm_endpoint=$(physwarm_extern_endpoint) || return 1

    # Check for an already-active Webots external controller.
    if [ -e "$physwarm_endpoint" ]; then
        echo "Webots external controller endpoint already exists:" >&2
        echo "  $physwarm_endpoint" >&2
        echo "Close the existing Webots instance, or run this launcher with 'existing'." >&2
        return 1
    fi

    # Keep the original TCP check as an additional guard.
    if command -v lsof >/dev/null 2>&1 && \
       lsof -nP -iTCP:"$physwarm_port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Webots controller port $physwarm_port is already in use." >&2
        echo "Close the existing Webots instance, or run this launcher with 'existing'." >&2
        return 1
    fi
}

wait_for_physwarm_webots() {
    physwarm_endpoint=$(physwarm_extern_endpoint) || return 1
    physwarm_elapsed=0

    echo "Waiting for Webots external controller..."

    while [ ! -e "$physwarm_endpoint" ]; do
        # Detect Webots crashing during startup.
        if [ -n "${physwarm_webots_pid:-}" ] && \
           ! kill -0 "$physwarm_webots_pid" 2>/dev/null; then
            echo "Webots exited before the external controller became ready." >&2

            if [ -n "${physwarm_webots_log:-}" ] && \
               [ -f "$physwarm_webots_log" ]; then
                echo "--- Webots log ---" >&2
                cat "$physwarm_webots_log" >&2
                echo "--- End Webots log ---" >&2
            fi

            return 1
        fi

        # Do not let the Python controller enter Webots while the
        # <extern> endpoint has not yet been registered.
        if [ "$physwarm_elapsed" -ge "$PHYSWARM_WEBOTS_STARTUP_TIMEOUT" ]; then
            echo "Timed out waiting for Webots external controller:" >&2
            echo "  $physwarm_endpoint" >&2

            if [ -n "${physwarm_webots_log:-}" ] && \
               [ -f "$physwarm_webots_log" ]; then
                echo "--- Webots log ---" >&2
                cat "$physwarm_webots_log" >&2
                echo "--- End Webots log ---" >&2
            fi

            return 1
        fi

        sleep 1
        physwarm_elapsed=$((physwarm_elapsed + 1))
    done

    echo "Webots external controller is ready:"
    echo "  $physwarm_endpoint"
}

start_physwarm_webots() {
    physwarm_mode=$1
    physwarm_repo_root=$2

    physwarm_webots_pid=""
    physwarm_webots_log=""

    ensure_physwarm_controller_url_valid || return 1

    case "$physwarm_mode" in
        existing)
            physwarm_endpoint=$(physwarm_extern_endpoint) || return 1

            if [ ! -e "$physwarm_endpoint" ]; then
                echo "No running Webots external controller found at:" >&2
                echo "  $physwarm_endpoint" >&2
                echo "Start Webots, run the simulation, and then retry with 'existing'." >&2
                return 1
            fi

            echo "Using existing Webots external controller:"
            echo "  $physwarm_endpoint"
            return 0
            ;;

        fast|slow)
            ensure_physwarm_controller_port_available || return 1

            physwarm_webots_bin="$WEBOTS_HOME/Contents/MacOS/webots"
            physwarm_world="$physwarm_repo_root/worlds/generated_world.wbt"

            if [ ! -x "$physwarm_webots_bin" ]; then
                echo "Webots executable not found:" >&2
                echo "  $physwarm_webots_bin" >&2
                return 1
            fi

            if [ ! -f "$physwarm_world" ]; then
                echo "Webots world not found:" >&2
                echo "  $physwarm_world" >&2
                return 1
            fi

            # Portable macOS/Linux temporary log file.
            physwarm_webots_log=$(
                mktemp "${TMPDIR:-/tmp}/physwarm-webots.XXXXXX"
            ) || {
                echo "Failed to create a Webots log file." >&2
                return 1
            }

            if [ "$physwarm_mode" = "fast" ]; then
                "$physwarm_webots_bin" \
                    --batch \
                    --mode=fast \
                    --no-rendering \
                    --stdout \
                    --stderr \
                    --extern-urls \
                    "$physwarm_world" \
                    >"$physwarm_webots_log" 2>&1 &

                physwarm_webots_pid=$!

                echo "Webots fast mode starting in background"
                echo "  PID: $physwarm_webots_pid"
                echo "  Log: $physwarm_webots_log"
            else
                "$physwarm_webots_bin" \
                    --mode=realtime \
                    --stdout \
                    --stderr \
                    --extern-urls \
                    "$physwarm_world" \
                    >"$physwarm_webots_log" 2>&1 &

                physwarm_webots_pid=$!

                echo "Webots realtime GUI starting"
                echo "  PID: $physwarm_webots_pid"
                echo "  Log: $physwarm_webots_log"
            fi

            # Critical synchronization point:
            # don't return to train_mappo.sh until Webots has actually
            # registered the <extern> supervisor.
            if ! wait_for_physwarm_webots; then
                stop_physwarm_webots
                return 1
            fi
            ;;

        *)
            echo "Usage: $0 [existing|slow|fast] [extra train_prey.py options]" >&2
            return 64
            ;;
    esac
}

stop_physwarm_webots() {
    if [ -n "${physwarm_webots_pid:-}" ]; then
        if kill -0 "$physwarm_webots_pid" 2>/dev/null; then
            kill "$physwarm_webots_pid" 2>/dev/null || true
            wait "$physwarm_webots_pid" 2>/dev/null || true
        fi

        physwarm_webots_pid=""
    fi
}