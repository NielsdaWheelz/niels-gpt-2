# Interfaces

Single source of truth for module boundaries + data shapes. If code disagrees with this doc, the code is wrong.

## Notation

- **V**: vocab size = 256 (byte tokens)
- **B**: batch size (runtime-chosen; must fit mps)
- **T**: context length = 256 (tokens per training example)
- **C**: model width (d_model) = 256
- **L**: number of transformer blocks = 4
- **H**: attention heads = 4
- **D**: head dim = C / H = 64 (must be even for RoPE)

### Dtypes

- **token ids**: `torch.int64`
- **activations / logits**: `torch.float32` (v0)

---

## Tokenizer

### Contract

```python
def encode(text: str) -> torch.LongTensor:
    """
    returns: ids shape (n,), dtype int64, values in [0..255]
    rule: utf-8 encode to bytes; each byte is one token id
    """

def decode(ids: torch.LongTensor) -> str:
    """
    inverse-ish of encode for display
    rule: bytes(ids.tolist()).decode("utf-8", errors="replace")
    """
```

### Invariants

- `decode(encode(s)) == s` is not guaranteed for arbitrary unicode, because the model can emit invalid utf-8 byte sequences; we display with `errors="replace"`.

---

## Chat Transcript Format

### Purpose

Define a text format for chat that the model learns to continue.

### Format Specification

**Exact format:**
- Each turn: `{role}: {content}\n` where role ∈ {system, user, assistant}
- Role tags must appear at start of line (position 0 or immediately after `\n`)
- Generation prompts end with exactly `assistant: ` (one space, no newline)

The trailing space after `assistant:` ensures the first generated token doesn't start glued to the colon.

**Persona constraint (training + inference):**
- Assistant speaks in third person about niels (e.g. "niels is…", not "i am…").

### Formatting Contract

```python
def format_chat(messages: list[dict]) -> str:
    """
    messages: list of dicts with keys "role" and "content"
      role ∈ {"system", "user", "assistant"}

    returns: formatted transcript string ending with "assistant: "

    exact format per message: "{role}: {content}\n"
    final line: "assistant: " (one space, no newline)
    """
```

### Parsing Contract

```python
def extract_assistant_reply(generated_text: str) -> str:
    """
    generated_text includes the prompt prefix.
    return text after the final "assistant: " up to:
      - end of string, or
      - the next turn tag if it appears.

    turn tag patterns (must match exactly):
      - "\nsystem: " or "\nuser: " or "\nassistant: "
      (lowercase role, space after colon)

    invariants:
    - stop only at turn tags that follow \n (not mid-content)
    - role tags must be lowercase and exact (reject "User:" or missing space)
    """
```

---

## Datasets

### Wikitext

**Dataset ID:** `"wikitext"`
**Config:** `"wikitext-103-raw-v1"`
**Field:** `"text"`
**License:** CC-BY-SA-3.0 and GFDL (as per HF dataset card)

**Split strategy:** Uses HuggingFace-provided train/validation/test splits (not byte-split locally)

Other configs exist (`wikitext-103-v1`, `wikitext-2-raw-v1`, `wikitext-2-v1`) but are not used in v0.

Note: wikitext-103 is large (generated dataset size ~549MB per HF, higher on disk). Consider subset mode for faster iteration.

```python
from datasets import load_dataset

def load_wikitext() -> dict[str, list[str]]:
    """
    returns dict with keys: "train", "validation", "test"
    each value: list[str] of the 'text' field from that split

    note: repo standardizes on "train"|"val" naming; loader maps hf "validation" -> "val"
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    out = {}
    out["train"] = ds["train"]["text"]
    out["val"] = ds["validation"]["text"]
    out["test"] = ds["test"]["text"]
    return out
```

### Roam (personal corpus)

**Split strategy:** Split locally by document (file path), not by bytes

```python
def list_roam_paths(root_dir: str) -> list[str]:
    """
    recursively list all *.md file paths under root_dir.
    return list[str] of absolute paths, in deterministic order (sorted).
    """

def load_texts(paths: list[str]) -> list[str]:
    """
    read files at given paths as utf-8 text.
    return list[str], one per file, in same order as paths.
    """

def split_roam_paths(
    paths: list[str],
    *,
    val_frac: float = 0.1,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    split roam file paths into train/val by document.

    returns: (train_paths, val_paths)

    uses torch.randperm with seed for deterministic split.
    ensures val set is held-out from training.
    """
```

### Primer (chat bias corpus)

A plain text file `data/primer.txt` containing repeated dialogue blocks using the chat transcript format.

**Recommended delimiter:** `\n\n<dialogue>\n\n` (literal string) between dialogues.

**Split strategy:** Split locally by dialogue blocks, not by bytes

```python
def load_primer_text(path: str) -> str:
    """returns raw text exactly as stored (no cleaning)."""

def split_primer_dialogues(
    text: str,
    *,
    val_frac: float = 0.1,
    seed: int,
) -> tuple[str, str]:
    """
    split primer dialogues into train/val by dialogue blocks.

    returns: (train_text, val_text)

    split on "\n\n<dialogue>\n\n" delimiter.
    uses torch.randperm with seed for deterministic split.
    ensures val dialogues are held-out from training.
    """
```

---

## Byte Streams

### Contract

Each data source is converted to a single byte stream:

```python
def build_wiki_streams(docs_train: list[str], docs_val: list[str]) -> tuple[bytes, bytes]:
    """
    returns: (wiki_train_bytes, wiki_val_bytes)
    - encoding: utf-8
    - separator: exactly "\n\n" between documents
    """

def build_roam_streams(docs_train: list[str], docs_val: list[str]) -> tuple[bytes, bytes]:
    """
    returns: (roam_train_bytes, roam_val_bytes)
    - encoding: utf-8
    - separator: exactly "\n\n" between documents
    """

def build_primer_streams(text_train: str, text_val: str) -> tuple[bytes, bytes]:
    """
    returns: (primer_train_bytes, primer_val_bytes)
    - encoding: utf-8
    """
```

### Invariants

- Stream type: `bytes`
- Encoding: utf-8
- Separator: exactly `"\n\n"` between documents
- Batching may sample mid-utf-8 sequence (we train on bytes, not characters)

---

## Batching

### Contract

```python
def get_batch(
    sources: dict[str, bytes],
    *,
    p: dict[str, float],
    B: int,
    T: int = 256,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    sources: dict mapping source name -> byte stream
      expected keys depend on p, typically: "wiki", "roam", "primer"

    p: sampling probabilities per source (must sum to 1.0)
      example: {"wiki": 0.80, "roam": 0.19, "primer": 0.01}

    generator: optional torch.Generator for reproducibility
      if None, uses torch default generator

    for each of B items:
      - select source using torch.multinomial with probabilities p
      - sample random start index i in [0, len(source) - (T+1)] using torch.randint
      - take chunk = source[i : i+T+1]
      - x = chunk[:-1], y = chunk[1:]

    returns:
      x: (B, T) int64 on device
      y: (B, T) int64 on device

    invariant: y[:, j] == x[:, j+1] for all j in [0, T-2]

    reproducibility:
      - uses only torch RNG (no python random or numpy)
      - deterministic given generator seed
    """
```

### Notes

- Caller is responsible for providing correct train/val streams per source
- Each batch item independently samples a source and position (chunk-based, not token interleaving)
- v0 default probabilities (training): `{"wiki": 0.80, "roam": 0.19, "primer": 0.01}`
- v0 default probabilities (validation): `{"wiki": 1.00}` (roam/primer eval use separate eval streams)

---

## Model

### Model Interface

```python
class GPT(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        config: dict with keys from ModelConfig schema
          required: V, T, C, L, H, D, d_ff, dropout
        """

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        x: (B, T) int64, token ids in [0..255]
        returns logits: (B, T, V) float32
        """
```

### Architecture Invariants (v0)

- Decoder-only transformer, L blocks
- Token embedding: `nn.Embedding(V, C)` with shape (V, C)
- Pre-norm residual blocks
- Multi-head causal self-attention: H heads, D = C / H
- Attention scores scaled by `1/sqrt(D)` before softmax
- MLP hidden size: d_ff (typically 4*C)
- Dropout: applied (rate specified in model config, not interface)
- Output head: `nn.Linear(C, V, bias=False)` with shape (V, C)
- Weight tying: `lm_head.weight = tok_emb.weight` (no transpose; shapes match)

**Causal masking invariant:**
For each position t, attention may only use keys at positions ≤ t.

---

## RoPE

RoPE modifies attention by applying a position-dependent rotation to q and k before computing attention scores.

### Precomputed Cache

```python
def rope_cache(
    T: int,
    D: int,
    *,
    theta: float = 10000.0,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    precompute sin/cos tables for RoPE

    returns: (sin, cos) each shape (1, 1, T, D//2)
    theta: base for frequency computation (10000.0)

    cache is built for positions 0..T-1
    shape includes broadcast dims for (B, H, T, D) attention tensors

    if device/dtype are None, defaults to cpu/float32
    """
```

### Application

```python
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B, H, T, D)
    sin, cos: (1, 1, T, D//2) exactly (as returned by rope_cache)

    returns: q_rot, k_rot with same shapes/dtypes as inputs

    constraints:
    - apply rotation only to q and k (not v)
    - rotate across all D dimensions (pairwise rotation)
    - D must be even
    - shape must be preserved exactly

    invariants:
    - sin.dtype == q.dtype
    - sin.device == q.device
    - cos.dtype == q.dtype
    - cos.device == q.device
    """
```

---

## Loss

### Contract

```python
def loss_fn(logits: torch.FloatTensor, y: torch.LongTensor) -> torch.FloatTensor:
    """
    logits: (B, T, V)
    y: (B, T)

    computes cross-entropy over all B*T positions.
    returns scalar loss.
    """
```

---

## Training Loop

### Reproducibility

```python
def set_seed(seed: int) -> None:
    """
    sets torch RNG seed:
    - torch.manual_seed(seed)

    must be called before data loading and model init for reproducibility

    note: this repo uses ONLY torch RNG for all randomness
    (no python random module, no numpy random)
    """
```

**Invariants:**
- All randomness uses torch RNG only (batching, splits, evaluation)
- Deterministic mode is NOT required (nondeterministic ops allowed for performance)

### Optimizer

AdamW

### LR Schedule

Warmup + cosine decay.

**Formula:**

```python
def lr_at_step(step: int, total_steps: int) -> float:
    """
    step in [0, total_steps-1]

    - linear warmup for first warmup_steps
    - cosine decay from base_lr down to min_lr after warmup
    """
```

**v0 defaults (not interfaces):**
- `base_lr = 3e-4`
- `warmup_steps = 200`
- `min_lr = 3e-5` (base_lr * 0.1)

### Other Invariants

- Gradient clip global norm = 1.0
- Precision: float32
- Training uses fixed `total_steps` (decided in training config), not wall-clock timing

### Device Selection

```python
def get_device() -> str:
    """
    returns: "mps" if available on Apple Silicon, else "cpu"
    """
```

---

## Checkpoints

### Contract

```python
def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    """
    saves a checkpoint containing:
    - model.state_dict()
    - optimizer.state_dict()
    - step (int)
    - config (dict snapshot: V, T, C, L, H, etc.)

    path: .pt or .pth file
    """

def load_checkpoint(path: str, device: str) -> tuple[dict, dict, dict | None, int]:
    """
    loads a checkpoint

    returns: (config, model_state_dict, optimizer_state_dict_or_none, step)

    config: dict with model sizing params
    model_state_dict: dict from torch
    optimizer_state_dict: dict from torch, or None if not saved
    step: int

    config returned first so caller can reconstruct model architecture.
    """
```

### Invariants

- Checkpoints are publishable (no private data beyond model weights)
- Config dict must contain all model sizing params needed to reconstruct architecture
- Config is serialized as JSON alongside checkpoint (e.g., `checkpoint.pt` + `config.json`)
- Checkpoint format is PyTorch's standard `torch.save()` dict structure

---

## Evaluation

### Multiple Eval Streams

Training uses mixed sources; evaluation uses separate held-out streams to measure different capabilities.

**Eval splits:**
- `val_wiki`: general language modeling (wikitext val split)
- `val_roam`: personal corpus style (held-out roam docs via `split_roam_docs`)
- `val_chat`: chat capability (held-out primer dialogues via `split_primer_dialogues`)

```python
def evaluate(
    model: GPT,
    eval_streams: dict[str, bytes],
    *,
    eval_steps: int,
    B: int,
    T: int,
    device: str,
    generator: torch.Generator,
) -> dict[str, float]:
    """
    eval_streams: dict mapping eval name -> byte stream
      example keys: "val_wiki", "val_roam", "val_chat"

    generator: torch.Generator for deterministic sampling (required)

    for each stream:
      - sample eval_steps batches from that stream using generator
      - compute average loss

    returns: dict mapping stream name -> average loss
    """
```

**Notes:**

Two evaluation modes:

1. **Training-time validation (fast):**
   - Runs periodically during training
   - Uses only `val_wiki` stream
   - Measures general LM capability

2. **Full evaluation (optional, slower):**
   - Runs separately on all held-out streams: `val_wiki`, `val_roam`, `val_chat`
   - Measures capability on each domain separately
   - All val sets must be truly held-out (wikitext via HF, roam/primer split by doc/dialogue)

---

## Generation

### Contract

```python
def generate(
    model: GPT,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.9,
    top_k: int | None = 50,
) -> str:
    """
    - encode prompt_text -> ids
    - for each generation step:
      - if len(ids) > T: crop to last T tokens before forward pass
      - run forward pass on cropped ids
      - sample next token from last position logits (temperature + optional top-k)
      - append to ids
    - decode full generated sequence -> text

    cropping behavior:
    - input ids are cropped to last T tokens before each forward
    - RoPE uses positions within the provided window (0..len-1)
    - cropping preserves relative positions inside the window
    - information beyond the window is discarded
    """
```

---

## Configuration Schema

Config values are not hard interfaces (can be tuned), but must be specified for reproducibility.

### Model Config

```python
ModelConfig = {
    "V": int,        # vocab size (256 for byte-level)
    "T": int,        # context length
    "C": int,        # model width (d_model)
    "L": int,        # number of transformer blocks
    "H": int,        # attention heads
    "D": int,        # head dim (C / H, must be even for RoPE)
    "d_ff": int,     # MLP hidden size (typically 4*C)
    "dropout": float # dropout rate
}
```

### Training Config

```python
TrainConfig = {
    "total_steps": int,
    "base_lr": float,
    "warmup_steps": int,
    "min_lr": float,
    "seed": int,
    "B": int,  # batch size
    "p_train": dict[str, float],  # source sampling probs for training
    "p_val": dict[str, float],    # source sampling probs for validation
}
```

### v0 Defaults

**Model:**
- V=256, T=256, C=256, L=4, H=4, D=64, d_ff=1024, dropout=0.1

**Training:**
- base_lr=3e-4, warmup_steps=200, min_lr=3e-5
- p_train = {"wiki": 0.80, "roam": 0.19, "primer": 0.01}
- p_val = {"wiki": 1.00}
