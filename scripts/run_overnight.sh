#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Overnight Training Runner
# Optimized for ~8 hour runs on laptop (MPS/CPU)
# Skips tokenization and caching (assumes already done)
################################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Use production profile for full-scale training
PROFILE="pipeline-prod"
DEVICE="${DEVICE:-mps}"  # Default to MPS for Apple Silicon

echo "========================================"
echo "Overnight Training Pipeline"
echo "========================================"
echo "Profile: $PROFILE (prod pretrain → prod SFT)"
echo "Device: $DEVICE"
echo "Estimated time: 6-10 hours (depends on hardware)"
echo ""
echo "This will run:"
echo "  1. Pretrain: 10K steps, V=50257, T=512, C=384, L=8"
echo "  2. SFT: 10K steps, using best pretrain checkpoint"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
echo "========================================"
sleep 5

START_TIME=$(date +%s)
START_DATE=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "Started at: $START_DATE"
echo ""

# Run the pipeline
python -m train.run --phase pipeline --profile "$PROFILE" --device "$DEVICE"

END_TIME=$(date +%s)
END_DATE=$(date "+%Y-%m-%d %H:%M:%S")
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "Training complete!"
echo "Started:  $START_DATE"
echo "Finished: $END_DATE"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
echo ""
echo "Checkpoints saved in: checkpoints/"
echo "Run details in: runs/<run_id>/resolved_settings.json"
echo ""
