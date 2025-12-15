### niels-gpt

tiny llm-from-scratch project + a small chat primer.

### quickstart

**with uv (recommended):**

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

**with pip:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

**device:** mac uses mps if available, otherwise cpu.

### what's implemented

**pr-01: scaffold + packaging + shared utilities**
- `niels_gpt.device` - device detection (mps/cpu)
- `niels_gpt.rng` - seeding and random number generation
- `niels_gpt.config` - dataclasses for ModelConfig and TrainConfig, json utilities
- `niels_gpt.paths` - repository path constants and ensure_dirs()

**pr-02: tokenizer + chat formatting**
- `niels_gpt.tokenizer` - byte-level encode/decode (utf-8 bytes as tokens)
- `niels_gpt.chat_format` - format_chat() and extract_assistant_reply()

### primer format

- **file**: `data/primer.txt`
- **turn format**: `system: …`, `user: …`, `assistant: …` (one line each)
- **dialogue delimiter**: literal `\n\n<dialogue>\n\n` between dialogues
- **system line**: identical in every dialogue (repetition is intentional)
- **no tag leakage**: assistant content should not contain `system:` / `user:` / `assistant:`

### generate "vast" dialogues (optional)

edit prompts + response patterns in `tools/primer_combinators.py`, then run:

```bash
python3 tools/generate_primer.py --seed 0 --n-per-category 30 --shuffle
```

this writes `data/primer.generated.txt`.
