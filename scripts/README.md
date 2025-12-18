# Training Scripts

Convenience scripts for running full training pipelines.

## Quick Start

### Overnight Training (Recommended for ~8hr runs)

Runs production-scale pretrain + SFT while you sleep:

```bash
./scripts/run_overnight.sh
```

**What it does:**
- Uses `pipeline-prod` profile (10K pretrain steps + 10K SFT steps)
- Full model: V=50257 (GPT-2 vocab), T=512, C=384, L=8, H=6
- Automatically chains pretrain → SFT using best checkpoint
- Defaults to MPS device (Apple Silicon)

**Requirements:**
- Tokenizer already trained at `artifacts/tokenizer/v2/spm.model`
- Caches already built in `data/cache/pretrain/` and `data/cache/sft/`

**Customize device:**
```bash
DEVICE=cpu ./scripts/run_overnight.sh
```

### Full Pipeline (First-time setup or data changes)

Runs everything from scratch: tokenization, caching, pretrain, SFT:

```bash
# First time: build everything
SKIP_TOKENIZER=false SKIP_CACHE=false ./scripts/run_full_pipeline.sh pipeline-prod

# Subsequent runs: skip setup
./scripts/run_full_pipeline.sh pipeline-prod

# Dev profile (faster, smaller models)
./scripts/run_full_pipeline.sh pipeline-dev
```

**What it does:**
1. **Tokenizer** (if `SKIP_TOKENIZER=false`): Trains SentencePiece tokenizer on your data
2. **Caches** (if `SKIP_CACHE=false`): Builds pretrain + SFT token caches
3. **Pretrain**: Runs pretrain phase using specified profile
4. **SFT**: Runs SFT phase using best pretrain checkpoint

**Environment variables:**
- `SKIP_TOKENIZER=false` - Rebuild tokenizer (default: true)
- `SKIP_CACHE=false` - Rebuild caches (default: true)
- `DEVICE=mps|cpu` - Override device selection (default: auto)

**Arguments:**
- `$1` - Profile name (default: `pipeline-dev`)
  - `pipeline-dev`: Small models, ~1K steps each (1-2 hours)
  - `pipeline-prod`: Full models, ~10K steps each (6-10 hours)

## Examples

```bash
# Quick dev test (~1-2 hours)
./scripts/run_full_pipeline.sh pipeline-dev

# Production run, assuming setup done
./scripts/run_overnight.sh

# Production run from scratch (first time)
SKIP_TOKENIZER=false SKIP_CACHE=false ./scripts/run_full_pipeline.sh pipeline-prod

# CPU-only overnight run
DEVICE=cpu ./scripts/run_overnight.sh

# Rebuild caches but keep tokenizer
SKIP_CACHE=false ./scripts/run_full_pipeline.sh pipeline-prod
```

## Output

All scripts output to:
- **Checkpoints**: `checkpoints/latest.pt`, `checkpoints/best.pt`
- **Run metadata**: `runs/<run_id>/resolved_settings.json`
- **Logs**: Console output (redirect to file if needed)

## Tips

**Monitor progress:**
```bash
# Run in background and log to file
./scripts/run_overnight.sh > training.log 2>&1 &

# Watch the log
tail -f training.log
```

**Resume from failure:**
```bash
# Training scripts use auto-resume by default
# Just re-run the same command, it will pick up from latest checkpoint
./scripts/run_overnight.sh
```

**Adjust for your hardware:**
- Edit profile configs in `configs/` to tune batch size, steps, etc.
- Use `--profile dev` for laptop-friendly runs
- Use `--profile prod` for GPU/overnight runs

## Profile Comparison

| Profile | Model Size | Steps | Est. Time | Use Case |
|---------|-----------|-------|-----------|----------|
| `pipeline-dev` | V=256, C=256, L=4 | 1K+1K | 1-2 hrs | Local iteration |
| `pipeline-prod` | V=50257, C=384, L=8 | 10K+10K | 6-10 hrs | Overnight GPU runs |

See all profiles: `python -m train.run --list-profiles`
