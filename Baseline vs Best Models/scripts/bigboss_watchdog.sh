LOG_DIR="$HOME/bigboss_baselines/logs"
SCRIPT_DIR="$HOME/bigboss_baselines/scripts"
RESULTS_DIR="$HOME/bigboss_baselines/results"
OFFLOAD_DIR="$HOME/bigboss_baselines/model_offload_baselines_ext4"
ENV_PATH="$HOME/bigboss-env-py312/bin/activate"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"

# check interval in minutes
INTERVAL_MINUTES=10

mkdir -p "$LOG_DIR"

echo " BigBoss Watchdog started at $(date)" >> "$WATCHDOG_LOG"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if python benchmark is running
    RUNNING=$(pgrep -f "model_runner_baseline.py" || true)

    if [[ -z "$RUNNING" ]]; then
        echo "[$TIMESTAMP]  Benchmark not running. Attempting resume..." >> "$WATCHDOG_LOG"

        # Clean up any stale offload data
        sudo rm -rf "$OFFLOAD_DIR"/* 2>/dev/null || true
        sudo sync

        # Activate environment and resume
        bash -c "
            source $ENV_PATH
            cd $SCRIPT_DIR
            MAX_NEW_TOKENS=12 ./run_all_baselines_final.sh >> $LOG_DIR/watchdog_resume_$(date +%Y%m%d_%H%M%S).log 2>&1
        " &
        echo "[$TIMESTAMP]  Resume triggered successfully." >> "$WATCHDOG_LOG"
    else
        echo "[$TIMESTAMP]  Benchmark still running (PID: $RUNNING)" >> "$WATCHDOG_LOG"
    fi

    sleep "${INTERVAL_MINUTES}m"
done
