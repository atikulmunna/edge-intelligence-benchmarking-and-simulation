set -euo pipefail
BASE=~/bigboss_baselines
MODEL_ID="meta-llama/Llama-3.1-8B"
PROMPTS="$BASE/configs/writing_prompts.json"
OFFLOAD="$BASE/model_offload_baselines_ext4/llama3_8b"
RESULTS="$BASE/results"
TOKENS="${MAX_NEW_TOKENS:-12}"

echo "Resuming BigBoss Baseline Benchmark for LLaMA 3.1 8B"
echo "--------------------------------------------------------"
echo "Model: $MODEL_ID"
echo "Offload dir: $OFFLOAD"
echo "Results: $RESULTS"
echo "Tokens: $TOKENS"
echo "--------------------------------------------------------"

CACHE_DIR="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
if [ ! -d "$CACHE_DIR" ]; then
  echo "Cached model not found! Run caching step first."
  exit 1
else
  echo "Cached model found at: $CACHE_DIR"
fi

# Run benchmark (resumable)
python3 "$BASE/scripts/model_runner_baseline.py" \
  --model "$MODEL_ID" \
  --prompts "$PROMPTS" \
  --offload_dir "$OFFLOAD" \
  --results_root "$RESULTS" \
  --max_new_tokens "$TOKENS" || {
    echo "Benchmark interrupted or failed."
    exit 1
  }

# Auto-compress and clean up
LATEST_RESULT=$(ls -td "$RESULTS"/* | head -1)
echo " Compressing latest results..."
zip -rq "${LATEST_RESULT}.zip" "$LATEST_RESULT"
echo " Results compressed to: ${LATEST_RESULT}.zip"

echo " Cleaning offload cache..."
rm -rf "$OFFLOAD" || true
echo " Offload cache cleared."

echo " LLaMA 3.1 8B benchmark completed successfully."
EOF
chmod +x ~/bigboss_baselines/scripts/resume_llama.sh
