#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Full Training Pipeline Runner
# Runs tokenization, caching, pretrain, and SFT in sequence
################################################################################

# Configuration
PROFILE="${1:-pipeline-dev}"  # Use pipeline-dev or pipeline-prod
DEVICE="${DEVICE:-auto}"       # Override with DEVICE=mps or DEVICE=cpu
SKIP_TOKENIZER="${SKIP_TOKENIZER:-true}"  # Set to false to retrain tokenizer
SKIP_CACHE="${SKIP_CACHE:-true}"          # Set to false to rebuild caches

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "Full Training Pipeline"
echo "========================================"
echo "Profile: $PROFILE"
echo "Device: $DEVICE"
echo "Skip tokenizer: $SKIP_TOKENIZER"
echo "Skip cache: $SKIP_CACHE"
echo "========================================"
echo ""

# Step 1: Tokenizer (optional, skip if already exists)
if [ "$SKIP_TOKENIZER" = "false" ]; then
    echo "==> Step 1/4: Training tokenizer"
    if [ ! -f "artifacts/tokenizer/v2/spm.model" ]; then
        python scripts/train_tokenizer.py \
            --input_glob ".roam-data/**/*.md" \
            --input_glob "data/primer.txt" \
            --include_wikitext \
            --fineweb_bytes 20000000 \
            --out_dir artifacts/tokenizer/v2 \
            --vocab_size 16000 \
            --seed 42
        echo "✓ Tokenizer trained"
    else
        echo "✓ Tokenizer already exists, skipping"
    fi
else
    echo "==> Step 1/4: Tokenizer (skipped)"
    if [ ! -f "artifacts/tokenizer/v2/spm.model" ]; then
        echo "⚠️  Warning: Tokenizer not found at artifacts/tokenizer/v2/spm.model"
        echo "   Set SKIP_TOKENIZER=false to train one"
    fi
fi
echo ""

# Step 2: Build caches (optional, skip if already exists)
if [ "$SKIP_CACHE" = "false" ]; then
    echo "==> Step 2/4: Building token caches"
    python -m niels_gpt.cache.cli build-all \
        --cache-dir data/cache \
        --roam-dir .roam-data \
        --fineweb-train-tokens 200000000 \
        --fineweb-val-tokens 5000000 \
        --shard-bytes 134217728 \
        --seed 42
    echo "✓ Caches built"
else
    echo "==> Step 2/4: Token caches (skipped)"
    if [ ! -d "data/cache/pretrain" ] || [ ! -d "data/cache/sft" ]; then
        echo "⚠️  Warning: Cache directories not found"
        echo "   Set SKIP_CACHE=false to build caches"
    fi
fi
echo ""

# Step 3 & 4: Run pipeline (pretrain → SFT)
echo "==> Step 3-4/4: Running training pipeline ($PROFILE)"
echo ""

START_TIME=$(date +%s)

if [ "$DEVICE" = "auto" ]; then
    python -m train.run --phase pipeline --profile "$PROFILE"
else
    python -m train.run --phase pipeline --profile "$PROFILE" --device "$DEVICE"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
