set -euo pipefail

BASE_DIR="$HOME/bigboss_baselines"
SCRIPTS_DIR="$BASE_DIR/scripts"
CONFIGS_DIR="$BASE_DIR/configs"
RESULTS_DIR="$BASE_DIR/results"
OFFLOAD_DIR_BASE="$BASE_DIR/model_offload_baselines_ext4"
LOG_DIR="$BASE_DIR/logs"
MAX_TOKENS="${MAX_NEW_TOKENS:-12}"
mkdir -p "$RESULTS_DIR" "$OFFLOAD_DIR_BASE" "$LOG_DIR"

# --------------------------------------------------------------------------
echo " Running BigBoss Baseline Safety Checks..."
[[ -d "$BASE_DIR" ]] || { echo " Base directory missing!"; exit 1; }
[[ -d "$CONFIGS_DIR" ]] || { echo " Configs folder missing!"; exit 1; }
[[ -d "$SCRIPTS_DIR" ]] || { echo " Scripts folder missing!"; exit 1; }
echo " Folder integrity verified."

FREE_GB=$(df -h / | awk 'NR==2 {print $4}' | sed 's/G//')
if (( ${FREE_GB%.*} < 10 )); then
  echo "  Low disk space (${FREE_GB} GB free). Consider cleanup before continuing."
fi

# --------------------------------------------------------------------------
#  Authentication
if ! hf auth whoami &>/dev/null; then
  echo " Logging into Hugging Face..."
  hf auth login --token "$HUGGINGFACE_TOKEN"
else
  echo " Auth confirmed."
fi

# --------------------------------------------------------------------------
#  Model map  (ID → prompt file + alias)
declare -A MODELS
#MODELS["google/gemma-3-4b-it"]="math_prompts.json gemma3_4b"
MODELS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]="code_prompts.json deepseek_qwen7b"
MODELS["meta-llama/Llama-3.1-8B"]="writing_prompts.json llama3_8b"
MODELS["HuggingFaceTB/SmolLM2-1.7B-Instruct"]="general_prompts.json smollm2_1p7b"

# --------------------------------------------------------------------------
#  Helper: expand swap dynamically
ensure_swap() {
  local swap_mb
  swap_mb=$(free -m | awk '/Swap/ {print $2}')
  if (( swap_mb < 3000 )); then
    echo " Expanding swap to 4 GB for heavy model safety..."
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
    sudo systemctl restart dphys-swapfile
    sleep 5
    echo "Swap expanded."
  fi
}

# --------------------------------------------------------------------------
#  Helper: download with infinite retry
download_with_retry() {
  local model_id="$1" attempt=1 sleep_time=60
  echo "Starting resilient download for $model_id"
  while true; do
    echo "[Attempt $attempt] ⏳ Downloading/caching..."
    python3 - <<EOF || true
from huggingface_hub import snapshot_download
try:
    snapshot_download("$model_id", token=True,
                      resume_download=True, max_workers=2,
                      local_dir_use_symlinks=False)
    print("Cached $model_id successfully.")
except Exception as e:
    print(f"Attempt failed: {e}")
    raise
EOF
    if [[ $? -eq 0 ]]; then break; fi
    echo " Retrying in $sleep_time s..."
    sleep "$sleep_time"
    (( attempt++ ))
    (( sleep_time = sleep_time < 600 ? sleep_time * 2 : 600 ))
  done
}

# --------------------------------------------------------------------------
#  Helper: cleanup + compress results
cleanup_after_model() {
  local offload_path="$1"
  echo " Cleaning offload cache: $offload_path"
  rm -rf "$offload_path" || true
  echo " Compressing latest results..."
  LAST_RESULT=$(ls -td $RESULTS_DIR/* | head -1)
  zip -rq "$LAST_RESULT.zip" "$LAST_RESULT"
  echo " Compressed: $LAST_RESULT.zip"
}

# --------------------------------------------------------------------------
#  Helper: run system diagnostics and log it
log_system_diagnostics() {
  local log_file="$1"
  {
    echo "===  BigBoss Diagnostic Snapshot ($(date)) ==="
    echo " Swap Status:" && free -h
    echo " Swapfile Config:" && grep CONF_SWAPSIZE /etc/dphys-swapfile
    echo " Temperature:" && vcgencmd measure_temp
    echo " Disk Usage:" && df -h / | awk 'NR==1 || NR==2'
    echo "==============================================="
  } >> "$log_file" 2>&1
}

# --------------------------------------------------------------------------
#  Main loop
for MODEL_ID in "${!MODELS[@]}"; do
  read -r PROMPT_FILE MODEL_NAME <<< "${MODELS[$MODEL_ID]}"
  PROMPTS_PATH="$CONFIGS_DIR/$PROMPT_FILE"
  OFFLOAD_DIR="$OFFLOAD_DIR_BASE/$MODEL_NAME"
  mkdir -p "$OFFLOAD_DIR"
  LOG_FILE="$LOG_DIR/${MODEL_NAME}_$(date +%Y%m%d%H%M%S).log"

  echo "---------------------------------------------"
  echo " Running: $MODEL_NAME ($MODEL_ID)"
  echo " Prompts: $PROMPTS_PATH"
  echo " Offload dir: $OFFLOAD_DIR"
  echo " Results: $RESULTS_DIR"
  echo " Tokens: $MAX_TOKENS"
  echo "---------------------------------------------"

  # Network guard
  until ping -c 1 huggingface.co &>/dev/null; do
    echo " Waiting for network..."
    sleep 60
  done

  # Heavy model check
  case "$MODEL_ID" in
    *"7B"*|*"8B"*)
      echo " Heavy model detected ($MODEL_ID) → Enabling FP16 offload mode."
      ensure_swap
      FP_MODE="--dtype float16"
      ;;
    *)
      FP_MODE="--dtype float32"
      ;;
  esac

  #  Log system diagnostics before run
  log_system_diagnostics "$LOG_FILE"

  # Safe download
  download_with_retry "$MODEL_ID"

  # Benchmark run
  echo "[INFO] Starting benchmark for $MODEL_NAME..."
  python3 "$SCRIPTS_DIR/model_runner_baseline.py" \
      --model "$MODEL_ID" \
      --prompts "$PROMPTS_PATH" \
      --offload_dir "$OFFLOAD_DIR" \
      --results_root "$RESULTS_DIR" \
      --max_new_tokens "$MAX_TOKENS" \
      $FP_MODE \
      >>"$LOG_FILE" 2>&1 || {
        echo " Benchmark error for $MODEL_NAME → check $LOG_FILE"
        continue
      }

  echo " Completed benchmark for $MODEL_NAME"
  cleanup_after_model "$OFFLOAD_DIR"
  echo "---------------------------------------------"
done

echo " All baseline benchmarks completed successfully!"