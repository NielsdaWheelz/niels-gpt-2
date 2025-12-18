# Configuration System Migration Guide

## Summary of Changes

The configuration system has been modernized with **named profiles** and improved CLI ergonomics. All functionality remains backward compatible through the `--config` flag.

## What Changed

### 1. Config File Naming (Clarity & Consistency)

Old config files have been renamed to follow a clear naming convention:

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `pretrain_smoke.json` | `smoke-pretrain.json` | Ultra-fast pretrain validation (2 steps) |
| `sft_smoke.json` | `smoke-sft.json` | Ultra-fast SFT validation (2 steps) |
| `train.json` | `dev-pretrain.json` | Development pretrain (1K steps) |
| `pretrain_overnight.json` | `prod-pretrain.json` | Production pretrain (10K steps, GPT-2 vocab) |
| `pretrain.json` | `minimal-pretrain.json` | Minimal pretrain config (mostly defaults) |
| `sft.json` | `minimal-sft.json` | Minimal SFT config (mostly defaults) |
| `smoke.json` | *removed* | Replaced by `dev-pretrain.json` |

**New configs added:**
- `dev-sft.json` - Development SFT config
- `prod-sft.json` - Production SFT config
- `pipeline-dev.json` - Development pipeline (dev pretrain → dev SFT)
- `pipeline-prod.json` - Production pipeline (prod pretrain → prod SFT)

### 2. Profile Registry System

A new **profile registry** was added at [niels_gpt/profiles.py](niels_gpt/profiles.py) that:
- Defines metadata for all available profiles
- Provides self-documenting descriptions
- Enables profile discovery and validation
- Groups profiles by purpose (smoke/dev/prod/minimal/pipeline)

### 3. Improved CLI

The training CLI ([train/run.py](train/run.py)) now supports:

**Named profiles** (recommended):
```bash
python -m train.run --phase pretrain --profile smoke-pretrain
python -m train.run --phase pretrain --profile dev
python -m train.run --phase sft --profile dev-sft
```

**Profile listing**:
```bash
python -m train.run --list-profiles
```

**Custom configs** (backward compatible):
```bash
python -m train.run --phase pretrain --config configs/my-custom.json
```

**Improved help text**:
```bash
python -m train.run --help
```

## Migration Guide

### If you were using old config names in scripts:

**Before:**
```bash
python -m train.run --phase pretrain --config configs/pretrain_smoke.json
python -m train.run --phase pretrain --config configs/train.json
python -m train.run --phase sft --config configs/sft.json
```

**After (Option 1: Use profiles - RECOMMENDED):**
```bash
python -m train.run --phase pretrain --profile smoke-pretrain
python -m train.run --phase pretrain --profile dev
python -m train.run --phase sft --profile dev-sft
```

**After (Option 2: Update config paths):**
```bash
python -m train.run --phase pretrain --config configs/smoke-pretrain.json
python -m train.run --phase pretrain --config configs/dev-pretrain.json
python -m train.run --phase sft --config configs/minimal-sft.json
```

### If you have custom configs that reference old files:

Update any pipeline configs or scripts that reference old config names:

**Before:**
```json
{
  "pretrain_config_path": "configs/pretrain.json",
  "sft_config_path": "configs/sft.json"
}
```

**After:**
```json
{
  "pretrain_config_path": "configs/minimal-pretrain.json",
  "sft_config_path": "configs/minimal-sft.json"
}
```

Or use the new pipeline profiles:
```bash
python -m train.run --phase pipeline --profile pipeline-dev
python -m train.run --phase pipeline --profile pipeline-prod
```

### If you created custom configs based on old examples:

Your custom configs continue to work via `--config path.json`. No changes needed.

For better integration with the new system, consider:
1. Using a profile as a base and only overriding what you need
2. Placing custom configs in `configs/` directory
3. Following the new naming convention: `{purpose}-{phase}.json`

## Available Profiles

Run `python -m train.run --list-profiles` to see all profiles. Current profiles:

**Smoke Tests** (ultra-fast validation):
- `smoke-pretrain` - 2 steps, tiny model (V=256, T=32, C=64, L=2)
- `smoke-sft` - 2 steps, standard SFT

**Development** (laptop-friendly):
- `dev` - 1K steps, small model (V=256, T=128, C=256, L=4)
- `dev-sft` - Development SFT with small model

**Production** (full-scale):
- `prod` - 10K steps, full model (V=50257, T=512, C=384, L=8, GPT-2 vocab)
- `prod-sft` - 10K steps, production SFT

**Minimal/Baseline** (starting points):
- `minimal` - Bare-bones pretrain config
- `minimal-sft` - Bare-bones SFT config

**Pipelines** (two-phase training):
- `pipeline-dev` - Development pipeline (dev pretrain → dev SFT)
- `pipeline-prod` - Production pipeline (prod pretrain → prod SFT)

## Benefits of the New System

1. **Discoverability**: `--list-profiles` shows all available options
2. **Self-documenting**: Each profile has a description and notes
3. **Type safety**: Profile names are validated at runtime
4. **Clarity**: Clear naming convention (smoke/dev/prod)
5. **Consistency**: Matching pairs (dev + dev-sft, prod + prod-sft)
6. **Backward compatibility**: `--config` still works for custom configs

## No Breaking Changes

All existing functionality is preserved:
- Custom config files via `--config` continue to work
- Settings resolution logic unchanged
- All Pydantic validation unchanged
- Resolved settings JSON output unchanged

The new system is purely additive - old workflows continue to work while new workflows are more ergonomic.
