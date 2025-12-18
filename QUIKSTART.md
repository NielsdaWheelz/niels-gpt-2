# QUICK START

### prerequisites
- python env (repo uses torch, datasets, sentencepiece). activate your venv and `pip install -e .` (or `pip install -r requirements.txt` if you keep one).
- repo paths: `data/.roam-data/` for roam markdown, `data/primer.txt` for primer text, tokenizer expected at `artifacts/tokenizer/v2/spm.model`. checkpoints land in `checkpoints/`.

### 1) prepare raw data
- roam: dump your roam export as markdown under `data/.roam-data/` (any nesting). default paths/flags expect that.
- primer: create `data/primer.txt`; separate dialogue blocks with the delimiter `\n\n<dialogue>\n\n` and avoid empty blocks. splitter rules are enforced here:

```4:63:niels_gpt/data/primer.py
DIALOGUE_DELIM = "\n\n<dialogue>\n\n"
...
def split_primer_dialogues(...):
    # drops empty blocks, deterministic shuffle, val_frac split
```

### 2) tokenizer (skip if `artifacts/tokenizer/v2/spm.model` exists)
- expected special ids: sys=3, usr=4, asst=5, eot=6 (see `niels_gpt/tokenizer.py` defaults).
- train a fresh one (feeds roam + primer + optional hf streams):

```bash
python scripts/train_tokenizer.py \
  --input_glob "data/.roam-data/**/*.md" \
  --input_glob "data/primer.txt" \
  --include_wikitext \
  --fineweb_bytes 20000000 \
  --out_dir artifacts/tokenizer/v2 \
  --vocab_size 16000 \
  --seed 42
```

### 3) build token caches
- bundled CLI builds fineweb-edu, roam, wikitext pretrain shards and dolly/oasst1 sft shards:

```bash
python -m niels_gpt.cache.cli build-all \
  --cache-dir data/cache \
  --roam-dir data/.roam-data \
  --fineweb-train-tokens 200000000 \
  --fineweb-val-tokens 5000000 \
  --shard-bytes 134217728 \
  --seed 42
```

flags/behavior:

```366:405:niels_gpt/cache/cli.py
parser.add_argument("command", choices=["build-all"])
parser.add_argument("--cache-dir", default="cache")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fineweb-train-tokens", type=int, default=200_000_000)
parser.add_argument("--fineweb-val-tokens", type=int, default=5_000_000)
parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
parser.add_argument("--roam-dir", type=str, default=None)
```

outputs (under `data/cache/`): `pretrain/{fineweb_edu,roam,wikitext}/(train|val)/shard_*.bin + meta.json`, `sft/{dolly15k,oasst1}/train/val bins+idx+meta`. no primer support in this CLI.

- primer cache (manual for now): build a pretrain shard yourself so `mix_pretrain` can reference `"primer"`:

```bash
python - <<'PY'
from pathlib import Path
from niels_gpt.cache.build_cache import build_pretrain_cache
from niels_gpt.data.primer import load_primer_text, split_primer_dialogues, DIALOGUE_DELIM
from niels_gpt.tokenizer import get_default_tokenizer
tok = get_default_tokenizer()
raw = load_primer_text("data/primer.txt")
train_txt, val_txt = split_primer_dialogues(raw, val_frac=0.1, seed=42, delimiter=DIALOGUE_DELIM)
val_tokens = len(tok.encode(val_txt))
train_tokens = len(tok.encode(train_txt))
build_pretrain_cache(
    [val_txt, train_txt],
    "data/cache/pretrain/primer",
    tokenizer=tok,
    max_train_tokens=train_tokens,
    max_val_tokens=val_tokens,
    shard_bytes=128_000_000,
    seed=42,
    shuffle_buffer=None,
    source_name="primer",
    source_config=None,
    streaming=False,
)
PY
```

naming matters: whatever key you put in `data.mix_pretrain` must match the directory under `data/cache/pretrain/`.

### 4) author configs (json overrides to the defaults in `niels_gpt/settings.py`)
- minimal pretrain override that blends fineweb + roam + primer (matching cache names):

```json
{
  "data": {
    "caches": { "pretrain_token_cache": "data/cache/pretrain" },
    "mix_pretrain": { "fineweb_edu": 0.6, "wikitext": 0.2, "roam": 0.1, "primer": 0.1 },
    "val_pretrain_source": "wikitext"
  },
  "training": { "pretrain": { "total_steps": 1000, "micro_B": 16, "accum_steps": 4, "amp": true, "amp_dtype": "fp16" } }
}
```

- sft override (uses dolly/oasst1 caches):

```json
{
  "data": { "caches": { "sft_token_cache": "data/cache/sft" } },
  "training": { "sft": { "micro_B": 8, "total_steps": 10000, "amp": true } }
}
```

- pipeline config file (just points to the two JSONs):

```json
{ "pretrain_config_path": "configs/minimal-pretrain.json", "sft_config_path": "configs/minimal-sft.json" }
```

### 5) run training
- CLI and flags:

```bash
# List available profiles
python -m train.run --list-profiles

# Use a named profile (recommended)
python -m train.run --phase pretrain --profile smoke-pretrain
python -m train.run --phase pretrain --profile dev
python -m train.run --phase sft --profile dev-sft

# Or use a custom config file
python -m train.run --phase pretrain --config configs/my-config.json
```

Available profiles:
- **smoke-pretrain**, **smoke-sft**: Ultra-fast validation (2 steps, ~30s on CPU)
- **dev**, **dev-sft**: Development iteration (small models, laptop-friendly)
- **prod**, **prod-sft**: Production-scale training (full models, GPU required)
- **minimal**, **minimal-sft**: Bare-bones configs with minimal overrides
- **pipeline-dev**, **pipeline-prod**: Two-phase pretrain→SFT

Example commands:
- pretrain: `python -m train.run --phase pretrain --profile dev --device mps`
- sft: `python -m train.run --phase sft --profile dev-sft --device mps --resume checkpoints/latest.pt` (or `--no-resume` to start fresh)
- pipeline: `python -m train.run --phase pipeline --profile pipeline-dev --device mps`

each run writes `runs/<run_id>/resolved_settings.json` plus checkpoints in `checkpoints/`.

### 6) chat / generation
- interactive CLI (greedy/top-k/top-p sampling) driven by your checkpoint and settings defaults:

```23:163:niels_gpt/chat_cli.py
parser.add_argument("--ckpt", required=True)
parser.add_argument("--max-new-tokens", default=gen_defaults.max_new_tokens)
parser.add_argument("--temperature", default=gen_defaults.temperature)
parser.add_argument("--top-k", default=gen_defaults.top_k or 0)
parser.add_argument("--top-p", default=gen_defaults.top_p or 0.0)
parser.add_argument("--seed", default=42)
parser.add_argument("--system", default=None)
parser.add_argument("--system-file", default=None)
...
generated_text = generate_text(...)
reply = extract_assistant_reply(generated_text)
```

run:

```bash
python -m niels_gpt.chat_cli \
  --ckpt checkpoints/best.pt \
  --max-new-tokens 256 \
  --temperature 0.9 \
  --top-k 50 \
  --top-p 0.0 \
  --seed 42 \
  --system-file configs/system_surly.txt   # optional
```

defaults: system prompt from file if present, otherwise a built-in surly voice; stop token is the eot id; banned tokens default to role tokens when `ban_role_tokens_during_generation` is true.

### 7) quick sanity
- fast cache + train smoke: use the smoke profiles (`--profile smoke-pretrain` and `--profile smoke-sft`). good for checking wiring and cache names before a long run.
- resolved settings in `runs/<run_id>/resolved_settings.json` tell you exactly what config was used (good for reproducibility).

### 8) overnight / full pipeline runs

convenience scripts for long-running training:

```bash
# Overnight run (~8 hours, production models)
# Assumes tokenizer and caches already exist
./scripts/run_overnight.sh

# Or customize device
DEVICE=cpu ./scripts/run_overnight.sh

# Full pipeline including tokenizer and cache building
# (only needed first time or when changing data)
SKIP_TOKENIZER=false SKIP_CACHE=false ./scripts/run_full_pipeline.sh pipeline-prod

# Dev pipeline (faster, for testing)
./scripts/run_full_pipeline.sh pipeline-dev
```

what each script does:
- `run_overnight.sh`: runs production pipeline (pretrain → SFT) with prod profiles. assumes setup already done.
- `run_full_pipeline.sh`: optionally builds tokenizer + caches, then runs pipeline. controlled via env vars:
  - `SKIP_TOKENIZER=false`: rebuild tokenizer
  - `SKIP_CACHE=false`: rebuild caches
  - `DEVICE=mps|cpu`: override device selection
