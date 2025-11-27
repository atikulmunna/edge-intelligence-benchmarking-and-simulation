set -euo pipefail

# === PATHS ===
BASE_DIR="$HOME/bigboss_baselines"
SCRIPTS_DIR="$BASE_DIR/scripts"
CONFIGS_DIR="$BASE_DIR/configs"
RESULTS_DIR="$BASE_DIR/results"
OFFLOAD_DIR_BASE="$BASE_DIR/model_offload_baselines_ext4"
LOG_DIR="$BASE_DIR/logs"
MAX_TOKENS="${MAX_NEW_TOKENS:-12}"
mkdir -p "$RESULTS_DIR" "$OFFLOAD_DIR_BASE" "$LOG_DIR"

# === SAFETY CHECKS ===
echo " Running BigBoss Baseline Safety Checks..."
[[ -d "$BASE_DIR" ]] || { echo " Base directory missing!"; exit 1; }
[[ -d "$CONFIGS_DIR" ]] || { echo " Configs folder missing!"; exit 1; }
[[ -d "$SCRIPTS_DIR" ]] || { echo " Scripts folder missing!"; exit 1; }
echo " Folder integrity verified."

FREE_GB=$(df -h / | awk 'NR==2 {print $4}' | sed 's/G//')
if (( ${FREE_GB%.*} < 10 )); then
  echo "Low disk space (${FREE_GB}GB free). Consider cleanup before continuing."
fi

# === AUTH CHECK ===
if ! hf auth whoami &>/dev/null; then
  echo " Logging into Hugging Face..."
  hf auth login --token "$HUGGINGFACE_TOKEN"
else
  echo " Auth confirmed."
fi

# === MODEL MAP ===
declare -A MODELS
MODELS["google/gemma-3-4b-it"]="math_prompts.json gemma3_4b"
MODELS["HuggingFaceTB/SmolLM2-1.7B-Instruct"]="general_prompts.json smollm2_1p7b"
MODELS["deepseek-ai/DeepSeek-Coder-6.7B-Instruct"]="json_prompts.json deepseekcoder"
MODELS["mistralai/Mistral-7B-Instruct-v0.3"]="writing_prompts.json mistral"
MODELS["meta-llama/Llama-3.1-8B"]="general_prompts.json llama3_8b"

# === RESUME SUMMARY ===
echo "---------------------------------------------"
echo " BigBoss Baseline Resume Summary"
echo "---------------------------------------------"

COMPLETED=()
PENDING=()

for model in gemma3_4b smollm2_1p7b deepseekcoder mistral llama3_8b; do
  if ls "$RESULTS_DIR" | grep -qi "$model"; then
    COMPLETED+=("$model")
  else
    PENDING+=("$model")
  fi
done

if [[ ${#COMPLETED[@]} -eq 0 ]]; then
  echo " No previous runs found — starting full benchmark."
else
  echo " Completed: ${COMPLETED[*]}"
fi

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo " All models already completed!"
  exit 0
else
  echo " Pending: ${PENDING[*]}"
fi
echo "---------------------------------------------"

# === OFFLOAD DIR CHECK ===
if [ "$(ls -A $OFFLOAD_DIR_BASE)" ]; then
  echo "Offload directory not empty. Cleaning up residual files..."
  sudo rm -rf "$OFFLOAD_DIR_BASE"/*
  echo " Cleaned $OFFLOAD_DIR_BASE"
fi

echo "---------------------------------------------"
echo " RESUME MODE ACTIVE — continuing with next pending model."
echo "---------------------------------------------"

# === FUNCTIONS ===

download_with_retry() {
  local model_id="$1"
  local attempt=1
  local max_attempts=9999
  local sleep_time=60

  echo " Starting resilient download for $model_id"

  while (( attempt <= max_attempts )); do
    echo "[Attempt $attempt] ⏳ Downloading/caching model..."
    python3 - <<EOF || true
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        "$model_id",
        token=True,
        resume_download=True,
        max_workers=2,
        local_dir_use_symlinks=False
    )
    print(" Successfully cached $model_id")
except Exception as e:
    print(f" Attempt failed: {e}")
    raise
EOF
    if [[ $? -eq 0 ]]; then
      echo " Model cached successfully for $model_id"
      break
    else
      echo "Download failed for $model_id (Attempt $attempt). Retrying in $sleep_time seconds..."
      sleep "$sleep_time"
      ((attempt++))
      (( sleep_time = sleep_time < 600 ? sleep_time * 2 : 600 ))
    fi
  done
}

cleanup_after_model() {
  local OFFLOAD_PATH="$1"
  echo " Cleaning offload cache: $OFFLOAD_PATH"
  rm -rf "$OFFLOAD_PATH" || true
  echo "Compressing latest results..."
  LAST_RESULT=$(ls -td $RESULTS_DIR/* | head -1)
  zip -rq "$LAST_RESULT.zip" "$LAST_RESULT"
  echo "Compressed: $LAST_RESULT.zip"

  FREE_SPACE=$(df -h / | awk 'NR==2 {print $4}')
  echo "Disk space remaining: $FREE_SPACE"
}

# === MAIN BENCHMARK LOOP ===
for MODEL_ID in "${!MODELS[@]}"; do
  read -r PROMPT_FILE MODEL_NAME <<< "${MODELS[$MODEL_ID]}"
  PROMPTS_PATH="$CONFIGS_DIR/$PROMPT_FILE"
  OFFLOAD_DIR="$OFFLOAD_DIR_BASE/$MODEL_NAME"
  mkdir -p "$OFFLOAD_DIR"

  # Skip if already benchmarked
  if ls "$RESULTS_DIR" | grep -qi "$MODEL_NAME"; then
    echo "Skipping $MODEL_NAME (already completed)."
    continue
  fi

  LOG_FILE="$LOG_DIR/${MODEL_NAME}_$(date +%Y%m%d%H%M%S).log"

  echo "---------------------------------------------"
  echo " Running: $MODEL_NAME ($MODEL_ID)"
  echo "Prompts: $PROMPTS_PATH"
  echo "Offload dir: $OFFLOAD_DIR"
  echo "Results: $RESULTS_DIR"
  echo "Tokens: $MAX_TOKENS"
  echo "---------------------------------------------"

  # Internet check
  until ping -c 1 huggingface.co &>/dev/null; do
    echo "Waiting for Internet connection..."
    sleep 60
  done

  # Download model with retry
  download_with_retry "$MODEL_ID"

  # Run benchmark
  echo "[INFO] Starting benchmark for $MODEL_NAME..."
  python3 "$SCRIPTS_DIR/model_runner_baseline.py" \
    --model "$MODEL_ID" \
    --prompts "$PROMPTS_PATH" \
    --offload_dir "$OFFLOAD_DIR" \
    --results_root "$RESULTS_DIR" \
    --max_new_tokens "$MAX_TOKENS" \
    >>"$LOG_FILE" 2>&1 || {
      echo "Benchmark error for $MODEL_NAME. Check log: $LOG_FILE"
      continue
    }

  echo "Completed benchmark for $MODEL_NAME"
  cleanup_after_model "$OFFLOAD_DIR"
  echo "---------------------------------------------"
done

echo "All baseline benchmarks completed successfully!"