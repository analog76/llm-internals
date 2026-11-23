#!/usr/bin/env bash
# serve-vllm.sh: run on the GPU instance (g5) launched from your saved AMI.
# Starts vLLM in a tmux session bound to 127.0.0.1 (reach it via SSH tunnel).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
VENV="$HOME/vllm-env"
LOG="$HOME/vllm-logs/serve-$(date +%Y%m%d-%H%M%S).log"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "No GPU found (nvidia-smi failed). Run this on the g5, not the build box." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "$HOME/vllm-logs"
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "source $VENV/bin/activate && \
  vllm serve $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_UTIL 2>&1 | tee $LOG"

echo "==> Waiting for vLLM on port $PORT (first run downloads the model, can take a few minutes)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "vLLM is up:"; curl -s "http://127.0.0.1:$PORT/v1/models"; echo
    echo "Logs: $LOG   Attach: tmux attach -t vllm   Metrics: curl localhost:$PORT/metrics"
    exit 0
  fi
  sleep 5
done
echo "vLLM did not come up in 10 minutes. Check: tail -n 50 $LOG" >&2
exit 1
