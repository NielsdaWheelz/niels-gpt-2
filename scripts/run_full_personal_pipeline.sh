#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Full personal pipeline runner (tokenizer + caches + pretrain + SFT)
# Config-driven; suitable for unattended overnight runs.
################################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"          # cuda | mps | cpu | auto
NO_RESUME="${NO_RESUME:-false}"   # set to true for a clean run
TOKENIZER_RETRIES="${TOKENIZER_RETRIES:-3}"
TOKENIZER_RETRY_DELAY="${TOKENIZER_RETRY_DELAY:-10}"
LOG_DIR="${LOG_DIR:-logs}"

PRETRAIN_CONFIG="configs/personal-pretrain.json"
SFT_CONFIG="configs/personal-sft.json"
PIPELINE_CONFIG="configs/pipeline-personal.json"

mkdir -p "$LOG_DIR"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/personal_pipeline_${RUN_TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

TOKENIZER_DIR="artifacts/tokenizer/v2"
TOKENIZER_MODEL="${TOKENIZER_DIR}/spm.model"
TOKENIZER_VOCAB="${TOKENIZER_DIR}/spm.vocab"
TOKENIZER_META="${TOKENIZER_DIR}/tokenizer_meta.json"

tokenizer_ready() {
  if [ ! -f "$TOKENIZER_MODEL" ] || [ ! -f "$TOKENIZER_VOCAB" ] || [ ! -f "$TOKENIZER_META" ]; then
    return 1
  fi
  python - <<'PY'
from niels_gpt.tokenizer import get_default_tokenizer
get_default_tokenizer()
print("tokenizer validation ok")
PY
}

echo "========================================"
echo "Full Personal Pipeline"
echo "========================================"
echo "Device: $DEVICE"
echo "No-resume: $NO_RESUME"
echo "Pretrain config: $PRETRAIN_CONFIG"
echo "SFT config: $SFT_CONFIG"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

if [ ! -f "data/primer.jsonl" ]; then
  echo "error: missing data/primer.jsonl (required for SFT primer cache)"
  exit 1
fi

if [ ! -f "data/primer.txt" ]; then
  echo "warning: data/primer.txt not found (tokenizer will skip it)"
fi

shopt -s nullglob globstar
roam_files=(data/.roam-data/**/*.md)
if [ ${#roam_files[@]} -eq 0 ]; then
  echo "error: no markdown found under data/.roam-data."
  echo "       add files there or remove 'roam' from mix_pretrain in $PRETRAIN_CONFIG"
  exit 1
fi

echo "==> Step 1/4: Train tokenizer"
if tokenizer_ready; then
  echo "✓ Tokenizer already present and valid; skipping"
else
  for attempt in $(seq 1 "$TOKENIZER_RETRIES"); do
    echo "tokenizer attempt ${attempt}/${TOKENIZER_RETRIES}"
    if python scripts/train_tokenizer.py \
      --input_glob "data/.roam-data/**/*.md" \
      --input_glob "data/primer.txt" \
      --include_wikitext \
      --fineweb_bytes 20000000 \
      --out_dir "$TOKENIZER_DIR" \
      --vocab_size 16000 \
      --seed 42; then
      echo "✓ Tokenizer trained"
      break
    fi
    if tokenizer_ready; then
      echo "✓ Tokenizer artifacts validate despite non-zero exit; continuing"
      break
    fi
    if [ "$attempt" -lt "$TOKENIZER_RETRIES" ]; then
      echo "tokenizer failed; retrying in ${TOKENIZER_RETRY_DELAY}s..."
      sleep "$TOKENIZER_RETRY_DELAY"
    else
      echo "error: tokenizer failed after ${TOKENIZER_RETRIES} attempts"
      exit 1
    fi
  done
fi
echo ""

echo "==> Step 2/4: Build token caches"
python -m niels_gpt.cache.cli build-all \
  --cache-dir data/cache \
  --roam-dir data/.roam-data \
  --fineweb-train-tokens 200000000 \
  --fineweb-val-tokens 5000000 \
  --shard-bytes 134217728 \
  --seed 42
echo "✓ Caches built"
echo ""

echo "==> Step 3-4/4: Run training pipeline"
TRAIN_ARGS=(--phase pipeline --config "$PIPELINE_CONFIG")
if [ "$DEVICE" != "auto" ]; then
  TRAIN_ARGS+=(--device "$DEVICE")
fi
if [ "$NO_RESUME" = "true" ]; then
  TRAIN_ARGS+=(--no-resume)
fi

python -m train.run "${TRAIN_ARGS[@]}"
echo ""
echo "✓ Pipeline complete"
