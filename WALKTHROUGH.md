# LLM From First Principles: A Complete Beginner's Guide

This document explains **everything** in this codebase from absolute zero. By the end, you'll understand how LLMs work, why every design choice was made, and what the alternatives are.

---

## Table of Contents

1. [What is an LLM?](#1-what-is-an-llm)
2. [The Tokenizer: Text → Numbers](#2-the-tokenizer-text--numbers)
3. [The Transformer Architecture](#3-the-transformer-architecture)
4. [Training: Teaching the Model](#4-training-teaching-the-model)
5. [Inference: Generating Text](#5-inference-generating-text)
6. [Configuration Reference](#6-configuration-reference)
7. [File-by-File Audit](#7-file-by-file-audit)
8. [Design Decisions and Alternatives](#8-design-decisions-and-alternatives)

---

## 1. What is an LLM?

### 1.1 The Core Idea

A Large Language Model (LLM) is a function that:
1. Takes a sequence of tokens (numbers representing text)
2. Outputs a probability distribution over what token comes next

That's it. The magic is in *how* it does this and *how* it learns.

### 1.2 Next-Token Prediction

Given: `"The cat sat on the "`
The model outputs probabilities: `{"mat": 0.3, "floor": 0.2, "chair": 0.15, ...}`

The model doesn't "understand" language - it's learned statistical patterns from billions of examples.

### 1.3 Key Symbols in This Codebase

| Symbol | Meaning | Default | Impact |
|--------|---------|---------|--------|
| **B** | Batch size | 16 | Memory usage, training stability |
| **T** | Context length (max tokens) | 512 | How much text the model can "see" |
| **V** | Vocabulary size | 16,000 | Number of unique tokens |
| **C** | Model width (embedding dim) | 384 | Model capacity, quality |
| **L** | Number of layers | 8 | Depth of processing |
| **H** | Number of attention heads | 6 | Parallel attention patterns |
| **D** | Head dimension (C/H) | 64 | Per-head capacity |
| **d_ff** | MLP hidden dimension | 1152 (3×C) | Feed-forward capacity |

---

## 2. The Tokenizer: Text → Numbers

### 2.1 Why We Need Tokens

Neural networks operate on numbers, not text. A tokenizer converts:
- `"Hello world"` → `[1234, 5678]` (encode)
- `[1234, 5678]` → `"Hello world"` (decode)

### 2.2 Tokenization Strategies

#### Byte-Level (Not Used Here)
- Each byte (0-255) is one token
- Pros: Simple, no OOV (out-of-vocabulary) words
- Cons: Very long sequences (8 bytes per character for some text)
- Verdict: Bad for transformers where attention is O(T²)

#### Subword (Used Here: SentencePiece Unigram)
- Common words are single tokens (`"the"` → `[42]`)
- Rare words are split (`"tokenization"` → `["token", "ization"]`)
- Pros: Shorter sequences, handles any text
- Cons: More complex training

#### Why Unigram over BPE?
- **BPE** (Byte-Pair Encoding): Merges most frequent pairs iteratively
- **Unigram**: Starts with large vocab, removes least useful tokens
- Unigram is more stable with diverse/multilingual data
- BPE is slightly faster to train
- This project chose unigram for mixed-domain robustness

### 2.3 Special Tokens

Special tokens mark structure in conversations:

```
<|ngpt_sys_84a5023f67d74cf29cc4001becde983c|>   - System message start
<|ngpt_usr_84a5023f67d74cf29cc4001becde983c|>   - User message start  
<|ngpt_asst_84a5023f67d74cf29cc4001becde983c|>  - Assistant message start
<|ngpt_eot_84a5023f67d74cf29cc4001becde983c|>   - End of turn
```

**Why the ugly UUIDs?** To avoid collision with real text. If someone types `<|system|>` in their message, it could break parsing. The UUID makes collision astronomically unlikely.

**Token IDs are hardcoded:** `sys=3, usr=4, asst=5, eot=6`. This is brittle but necessary - if IDs change, trained models break. The code validates these at load time.

### 2.4 Tokenizer Files

**`scripts/train_tokenizer.py`**:
- Trains SentencePiece with `user_defined_symbols` for special tokens
- Uses `byte_fallback=True` so unknown characters become byte sequences
- `character_coverage=1.0` means cover all characters in training data
- `split_digits=False` keeps numbers together
- `num_threads=1` for determinism (parallelism introduces randomness)

**Critique:** The script works but has no coverage evaluation. You trust downstream tests to catch problems. If you retrain with different specials, you must update `settings.TokenizerSettings.expected_special_token_ids`.

**`niels_gpt/tokenizer.py`**:
- Loads the trained model
- Validates each special token encodes to exactly one piece
- Validates decode(encode(special)) == special
- Provides `encode()` and `decode()` functions

**Critique:** Good safety checks. Slightly brittle if you want flexibility - IDs are fixed.

### 2.5 Vocab Size Tradeoffs

| V | Sequence Length | Embedding Cost | Use Case |
|---|-----------------|----------------|----------|
| 8K | Longer | Lower | Small models, limited data |
| 16K | Medium | Medium | **This project** |
| 32K | Shorter | Higher | Production models |
| 50K+ | Very short | Very high | GPT-4 class |

Larger vocab = shorter sequences = faster attention (O(T²)), but more embedding parameters (V × C).

---

## 3. The Transformer Architecture

### 3.1 High-Level Flow

```
Input: [token_ids]  shape (B, T)
    ↓
Token Embedding: lookup table V → C
    ↓
Dropout
    ↓
L × Transformer Block:
    ├─ RMSNorm
    ├─ Multi-Head Attention
    ├─ Residual + Dropout
    ├─ RMSNorm
    ├─ MLP (SwiGLU)
    └─ Residual + Dropout
    ↓
Final RMSNorm
    ↓
LM Head: C → V (tied with embedding)
    ↓
Output: [logits]  shape (B, T, V)
```

### 3.2 Embeddings

**What:** A lookup table where each token ID maps to a vector of size C.

```python
self.tok_emb = nn.Embedding(V, C)  # shape (V, C)
h = self.tok_emb(x)  # x: (B, T) → h: (B, T, C)
```

**Example:** If V=16000, C=384:
- Token 42 → vector of 384 numbers
- These numbers are *learned* during training

**Weight Tying:** The output projection (`lm_head`) shares weights with `tok_emb`:
```python
self.lm_head.weight = self.tok_emb.weight
```

Why? Two reasons:
1. **Fewer parameters:** Saves V×C parameters (~6M for our defaults)
2. **Better perplexity:** Forces input and output embeddings to be consistent

Alternative: Separate head (more params, sometimes better for huge vocab).

### 3.3 Positional Encoding: RoPE

**The Problem:** Attention is permutation-invariant. Without position info, "dog bites man" = "man bites dog".

**Old Solution (GPT-2):** Add learned position embeddings
- Pros: Simple
- Cons: Can't extrapolate beyond training T

**This Project: RoPE (Rotary Position Embedding)**

RoPE rotates the query and key vectors based on position:

```python
# For position p, rotate q/k by angle θ_i * p
# θ_i = θ^(-2i/D) where θ=10000 by default

# Even/odd components form pairs that get rotated
q_even_rot = q_even * cos(θ*p) - q_odd * sin(θ*p)
q_odd_rot  = q_even * sin(θ*p) + q_odd * cos(θ*p)
```

**Why RoPE?**
- Encodes *relative* position naturally (q·k depends on position difference)
- Extrapolates better than learned absolute
- No extra parameters

**rope_theta = 10000:** This is the base frequency. Higher theta = longer effective context. If you massively increase T, consider scaling theta.

**Implementation (`niels_gpt/model/rope.py`):**
- `rope_cache(T, D, theta)`: Precomputes sin/cos tables
- `apply_rope(q, k, sin, cos)`: Rotates q and k

**Critique:** Standard implementation. No bugs. Uses float32 for precision even when computing in fp16 via cached conversions.

### 3.4 Attention

Attention lets each position "look at" all previous positions and decide what's relevant.

**The Math:**
```
Q = x @ Wq   # queries: "what am I looking for?"
K = x @ Wk   # keys: "what do I contain?"
V = x @ Wv   # values: "what information do I provide?"

scores = (Q @ K^T) / sqrt(D)  # dot product similarity
scores = mask(scores)          # causal mask
weights = softmax(scores)      # normalize to probabilities
output = weights @ V           # weighted sum of values
```

**Causal Mask:** Position i can only attend to positions 0..i (not future). Without this, the model could "cheat" by looking at the answer.

```
Mask for T=4:
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

**Multi-Head Attention:**
Instead of one attention, compute H parallel attentions with D=C/H dimensions each:

```python
# Split C into H heads of D dimensions
q = q.reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
# ... compute attention per head ...
out = out.transpose(1, 2).reshape(B, T, C)  # merge heads
```

**Why multi-head?** Different heads can learn different patterns:
- Head 1: nearby words
- Head 2: syntactic relationships
- Head 3: long-range dependencies

**Implementation (`niels_gpt/model/blocks.py` - CausalSelfAttention):**
- Single `qkv` projection for efficiency: `(C → 3C)`
- RoPE applied after projection
- Causal mask registered as buffer (not recomputed)
- Dropout on attention weights and output

**Critique:** Clean implementation. Uses O(T²) naive attention - fine for T≤2048, would need FlashAttention for longer contexts.

### 3.5 MLP (Feed-Forward Network)

After attention mixes information across positions, MLP processes each position independently.

**Standard MLP:**
```
x → Linear(C, 4C) → ReLU → Linear(4C, C) → x
```

**This Project: SwiGLU**
```
gate = Linear(C, d_ff)(x)
value = Linear(C, d_ff)(x)
h = SiLU(gate) * value  # gating mechanism
out = Linear(d_ff, C)(h)
```

**Why SwiGLU over GELU/ReLU?**
- Better perplexity for similar FLOPs
- The gating mechanism helps information flow
- Used by LLaMA, Mistral, etc.

**d_ff = 1152 = 3 × C:** Rule of thumb is 4×C for standard MLP, but SwiGLU uses 2 parallel projections so 3×C ≈ equivalent cost.

**Critique:** Correct SwiGLU implementation. Biases are included (some implementations drop them for marginal speedup).

### 3.6 Normalization

**The Problem:** Deep networks have unstable gradients. Normalization keeps activations well-behaved.

**LayerNorm (Not Used):**
```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

**RMSNorm (Used Here):**
```
y = x / sqrt(mean(x²) + eps) * weight
```

**Why RMSNorm?**
- Simpler (no mean subtraction, no bias)
- Slightly faster
- Works just as well empirically

**Pre-norm vs Post-norm:**
- **Post-norm (original Transformer):** normalize after sublayer
- **Pre-norm (used here):** normalize before sublayer

Pre-norm is more stable for deep networks (gradients flow better through residuals).

**Implementation:**
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 3.7 Residual Connections

Each sublayer adds its input back:
```
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

**Why?** Gradients can flow directly through the residual path. Without residuals, gradients vanish in deep networks.

### 3.8 The Complete Block

```python
class Block(nn.Module):
    def forward(self, x):
        # Attention with pre-norm and residual (dropout inside attn)
        x = x + attn(norm1(x))
        # MLP with pre-norm and residual (dropout inside mlp)
        x = x + mlp(norm2(x))
        return x
```

### 3.9 Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)
```

**Why 0.02 std?** Empirically works well for transformer scale. Larger → unstable early training. Smaller → slow learning.

**Alternative:** Scaled initialization (e.g., divide by sqrt(2*L) for residual layers) for very deep models.

---

## 4. Training: Teaching the Model

### 4.1 The Training Objective

**Next-token prediction via cross-entropy loss:**

```python
# logits: (B, T, V) - model predictions
# targets: (B, T) - true next tokens

loss = F.cross_entropy(
    logits.view(-1, V),  # (B*T, V)
    targets.view(-1)      # (B*T,)
)
```

Cross-entropy loss penalizes the model for assigning low probability to the correct next token.

### 4.2 Two Training Phases

#### Pretraining
- Train on raw text (Wikipedia, web data, etc.)
- Objective: predict any next token
- Result: general language understanding

#### Supervised Fine-Tuning (SFT)
- Train on conversation examples
- Objective: predict only assistant responses
- Result: instruction-following behavior

**Loss Masking in SFT:**
```python
# Only compute loss on assistant tokens
# ignore_index=-100 tells PyTorch to skip these positions
y_masked[non_assistant_positions] = -100
loss = F.cross_entropy(logits, y_masked, ignore_index=-100)
```

### 4.3 Data Loading

**Pretrain Data (`train/pretrain.py`):**
- Memory-mapped shards (don't load all data into RAM)
- Random windows of length T+1 from each shard
- Mixture sampling across sources (fineweb: 70%, wikitext: 20%, roam: 10%)

```python
class PretrainSource:
    def sample(self, device, generator):
        # Pick random shard, random start position
        window = shard[start : start + T + 1]
        x = window[:-1]  # input
        y = window[1:]   # target (shifted by 1)
        return x, y
```

**SFT Data (`niels_gpt/cache/sft_dataset.py`):**
- Packed conversations with role tokens
- Assistant-only loss masking
- Example boundary tracking via index file

### 4.4 Gradient Accumulation

**Problem:** Large batch sizes improve training but don't fit in memory.

**Solution:** Accumulate gradients over multiple mini-batches:

```python
optimizer.zero_grad()
for _ in range(accum_steps):
    x, y = get_batch(B=micro_B)  # smaller batch
    loss = model(x) / accum_steps  # scale loss
    loss.backward()  # accumulate gradients
optimizer.step()  # update weights once

# Effective batch = micro_B * accum_steps
```

### 4.5 Learning Rate Schedule

**Warmup + Cosine Decay:**

```python
def lr_at_step(step, total_steps, base_lr, warmup_steps, min_lr):
    if step < warmup_steps:
        # Linear warmup: 0 → base_lr
        return base_lr * step / warmup_steps
    
    # Cosine decay: base_lr → min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + cos(π * progress))
    return min_lr + (base_lr - min_lr) * cosine
```

**Why warmup?** Early gradients are noisy. High LR early → divergence. Warmup gives the model time to stabilize.

**Why cosine decay?** Smooth decay works better than step drops. The model continues learning at low LR without sudden changes.

### 4.6 Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    params,
    lr=base_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)
```

**Adam:** Adaptive learning rates per parameter using momentum and variance estimates.

**AdamW:** Decouples weight decay from gradient updates. Better regularization.

**betas=(0.9, 0.95):**
- β1=0.9: momentum decay (how much past gradients matter)
- β2=0.95: variance decay (adaptive step size stability)

**weight_decay=0.1:** L2 regularization - prevents large weights, improves generalization.

**Parameter Groups:**
```python
# Don't apply weight decay to norms/bias/embeddings
decay_params = [p for name, p in model.named_parameters() 
                if 'norm' not in name and 'bias' not in name]
no_decay = [p for name, p in model.named_parameters()
            if 'norm' in name or 'bias' in name]
```

### 4.7 Mixed Precision (AMP)

**Problem:** fp32 is slow on modern hardware.

**Solution:** Train in fp16 where possible:

```python
with torch.autocast(device_type="mps", dtype=torch.float16):
    logits = model(x)
    loss = F.cross_entropy(logits, y)
loss.backward()  # gradients computed in fp16
# optimizer step in fp32 for stability
```

**fp16 vs bf16:**
- **fp16:** More precision, narrower range (can overflow)
- **bf16:** Less precision, same range as fp32 (safer)
- MPS supports fp16; bf16 support is limited

**Critique:** This codebase uses AMP on MPS only, falls back to fp32 on CPU. No gradient scaler (relies on torch's autocast). Works fine for this scale.

### 4.8 Activation Checkpointing

**Problem:** Storing activations for backprop uses lots of memory.

**Solution:** Recompute activations during backward pass:

```python
if self.activation_checkpointing and self.training:
    h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
else:
    h = block(h)
```

**Tradeoff:**
- Memory: ~20-40% reduction
- Speed: ~20-30% slower (recompute forward pass)

Use when memory-bound with large models.

### 4.9 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients by scaling down if total gradient norm exceeds threshold.

---

## 5. Inference: Generating Text

### 5.1 Autoregressive Generation

Generate one token at a time:

```python
for _ in range(max_new_tokens):
    logits = model(ids)        # (1, t, V)
    next_logits = logits[0, -1]  # (V,) - last position
    next_token = sample(next_logits)
    ids.append(next_token)
    if next_token == eot_id:
        break
```

### 5.2 Sampling Strategies

**Temperature:**
```python
logits = logits / temperature
# temperature < 1: sharper (more deterministic)
# temperature > 1: flatter (more random)
# temperature = 0: argmax (greedy)
```

**Top-K Filtering:**
```python
# Keep only top-k tokens, set rest to -inf
top_k_logits = top_k_filter(logits, k=50)
probs = softmax(top_k_logits)
```

**Top-P (Nucleus) Filtering:**
```python
# Keep smallest set of tokens whose cumulative prob ≥ p
sorted_probs = sort(softmax(logits))
cumulative = cumsum(sorted_probs)
cutoff = first index where cumulative >= p
mask out all tokens below cutoff
```

**Repetition Penalty:**
```python
for token in seen_tokens:
    logits[token] /= repetition_penalty
# repetition_penalty > 1 discourages repetition
```

**Typical Settings:**
- Creative writing: temp=0.9, top_k=50, top_p=0.9
- Factual: temp=0.7, top_k=20
- Greedy (deterministic): temp=0

### 5.3 KV-Cache: Making Generation Fast

**Problem:** Naive generation recomputes the full forward pass for each token. Cost: O(T²) per token → O(T³) total.

**Solution:** Cache the Key and Value matrices from attention:

```python
# Prefill: process entire prompt once
for layer in layers:
    k, v = compute_kv(prompt)
    cache[layer].k[:prompt_len] = k
    cache[layer].v[:prompt_len] = v

# Decode: process one token at a time
for _ in range(max_new_tokens):
    q, k, v = compute_qkv(last_token)
    cache[layer].k[pos] = k
    cache[layer].v[pos] = v
    
    # Attention over full cache
    attn = attention(q, cache.k[:pos+1], cache.v[:pos+1])
```

**Speedup:** O(T) per token instead of O(T²). Essential for real-time chat.

**Cache Shape:** `(L, B, H, T_max, D)`
- L layers, B batch, H heads, T_max positions, D head dim

**Implementation (`niels_gpt/infer/kv_cache.py`):**
- `allocate_kv_cache()`: Create zero-initialized cache
- `prefill()`: Process prompt, fill cache
- `decode_step()`: Process one token, append to cache

**Important:** Keys are stored *after* RoPE rotation. This is critical - if you store pre-RoPE keys, position encoding breaks.

**Critique:** Clean implementation following the spec. Hard cap on T_max enforced with clear errors. No streaming/ring-buffer for very long contexts.

---

## 6. Configuration Reference

### 6.1 Model Config (`ModelConfig`)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `V` | 16000 | 8K-100K | Vocab size. Larger = shorter sequences, more embedding params |
| `T` | 512 | 256-8192 | Context length. O(T²) memory/compute in attention |
| `C` | 384 | 128-4096 | Hidden dimension. Quality ∝ C², cost ∝ C² |
| `L` | 8 | 4-96 | Layers. Depth improves reasoning, cost ∝ L |
| `H` | 6 | 4-64 | Attention heads. Should divide C evenly, D=C/H ≈ 64-128 |
| `d_ff` | 1152 | C*2-C*4 | MLP hidden dim. Usually 3-4× C |
| `dropout` | 0.1 | 0-0.3 | Regularization. Lower for large data, higher for small |
| `rope_theta` | 10000 | 10K-1M | RoPE base frequency. Scale up for very long T |

### 6.2 Train Config (`TrainConfig`)

| Parameter | Default (Pretrain) | Default (SFT) | Explanation |
|-----------|-------------------|---------------|-------------|
| `micro_B` | 16 | 16 | Batch size per forward pass |
| `accum_steps` | 1 | 1 | Gradient accumulation steps |
| `total_steps` | 17000 | 6000 | Training iterations |
| `base_lr` | 3e-4 | 1e-4 | Peak learning rate |
| `warmup_steps` | 340 | 120 | LR warmup period |
| `min_lr` | 3e-5 | 1e-5 | Final learning rate |
| `weight_decay` | 0.1 | 0.05 | L2 regularization |
| `grad_clip` | 1.0 | 1.0 | Gradient clipping threshold |
| `amp` | false | false | Mixed precision training |
| `amp_dtype` | fp16 | fp16 | AMP precision type |

### 6.3 Generation Config (`GenerationSettings`)

| Parameter | Default | Range | Use Case |
|-----------|---------|-------|----------|
| `max_new_tokens` | 256 | 1-T | Maximum response length |
| `temperature` | 0.9 | 0-2 | Sampling randomness |
| `top_k` | 50 | 1-V | Top-k filtering |
| `top_p` | None | 0-1 | Nucleus sampling |
| `repetition_penalty` | None | 1-2 | Discourage repetition |

### 6.4 Data Mix (`mix_pretrain`, `mix_sft`)

**Pretrain defaults:**
```json
{"fineweb_edu": 0.70, "wikitext": 0.20, "roam": 0.10}
```

**SFT defaults:**
```json
{"oasst1": 0.67, "dolly15k": 0.33}
```

Probabilities must sum to 1.0.

### 6.5 Scaling Guidelines

**Tiny (laptop):** T=512, C=384, L=8, H=6
- ~10M params, trains in hours
- Coherent for short prompts, limited knowledge

**Small:** T=1024, C=512, L=12, H=8
- ~50M params, trains in days
- Better reasoning, longer context

**Medium:** T=2048, C=768, L=16, H=12
- ~200M params, needs GPU
- Reasonable quality for many tasks

**Large:** T=4096, C=1024, L=24, H=16
- ~500M+ params, needs good GPU
- Approaching useful quality

---

## 7. File-by-File Audit

### 7.1 Core Model

#### `niels_gpt/model/gpt.py`
**Purpose:** Top-level GPT class that wires everything together.

**Architecture:**
```
tok_emb(V, C) → dropout → [Block × L] → ln_f → lm_head(C, V)
```

**Key features:**
- Weight tying: `lm_head.weight = tok_emb.weight`
- Activation checkpointing support
- Attention tracing for visualization

**Critique:** Clean implementation. T assertion prevents accidental overflow. Weight tying is correct. Dropout applied after embedding (standard).

**Issues:** None significant.

#### `niels_gpt/model/blocks.py`
**Purpose:** Transformer block, attention, MLP, and normalization.

**CausalSelfAttention:**
- Single QKV projection (efficient)
- RoPE cache per dtype (fp16/bf16/fp32)
- Causal mask as buffer (not recomputed)
- `prefill` and `decode_step` for KV-cache

**MLP (SwiGLU):**
```python
def forward(self, x):
    a = self.fc_a(x)           # value path
    g = F.silu(self.fc_g(x))   # gate path
    h = g * a                   # gating
    return self.fc_out(h)
```

**RMSNorm:**
```python
def forward(self, x):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return x / rms * self.weight
```

**Critique:** All implementations are correct. The RoPE dtype caching is smart (avoids repeated conversions). Block structure follows pre-norm pattern correctly. Dropout is applied exactly once per sublayer: attn_dropout + resid_dropout in attention, resid_dropout in MLP. No block-level dropout wrapper.

#### `niels_gpt/model/rope.py`
**Purpose:** RoPE implementation.

**`rope_cache(T, D, theta)`:**
- Computes sin/cos for all positions
- Returns (1, 1, T, D//2) tensors

**`apply_rope(q, k, sin, cos)`:**
- Splits q/k into even/odd pairs
- Applies rotation
- Reconstructs

**Critique:** Correct implementation. Handles MPS float64 limitation by using float32 for computation. Extensive shape validation.

### 7.2 Tokenizer

#### `niels_gpt/special_tokens.py`
**Purpose:** Define special tokens and collision detection.

```python
SENTINEL_UUID = "84a5023f67d74cf29cc4001becde983c"
SYS_TOKEN = f"<|ngpt_sys_{SENTINEL_UUID}|>"
```

**`find_special_collision(text)`:** Returns collision info if any special token appears in text.

**`assert_no_special_collision(text, ...)`:** Raises if collision found.

**Critique:** Good safety mechanism. The UUID approach is sound. Collision detection is O(n) per special token - fine for this scale.

#### `niels_gpt/tokenizer.py`
**Purpose:** Load and use SentencePiece tokenizer.

**Validation on load:**
1. Each special token exists in vocab
2. Each encodes to exactly one piece
3. Each decodes back correctly
4. IDs match expected values

**Critique:** Robust validation. The lazy loading via `_DEFAULT_TOKENIZER` singleton is fine. Error messages are clear.

#### `scripts/train_tokenizer.py`
**Purpose:** Train SentencePiece from text files and HF datasets.

**Key settings:**
- `user_defined_symbols` for special tokens
- `byte_fallback=True` handles unknown chars
- `num_threads=1` for determinism

**Critique:** Works correctly. Missing features: coverage evaluation, automatic vocab sizing. The seed doesn't actually control SentencePiece determinism fully (it's best-effort).

### 7.3 Data and Caching

#### `niels_gpt/cache/build_cache.py`
**Purpose:** Build token caches for pretrain and SFT.

**Pretrain cache:**
- Sharded .bin files (uint16)
- Val tokens first, then train
- Shuffle buffer for streaming

**SFT cache:**
- tokens.bin + idx.npy (offsets)
- Conversations split by randperm

**Critique:** Clean implementation. The val-first split is simple but could skew distributions with very long documents. The shuffle buffer is only applied when streaming.

#### `niels_gpt/cache/sft_dataset.py`
**Purpose:** Memory-mapped SFT dataset.

**Assistant-only masking:**
```python
def _assistant_mask(seq, asst_id, eot_id):
    mask = zeros(len(seq) - 1)
    in_assistant = False
    for i in range(len(seq) - 1):
        if seq[i] == asst_id:
            in_assistant = True
        if in_assistant:
            mask[i] = True
        if in_assistant and seq[i+1] == eot_id:
            in_assistant = False
    return mask
```

**Critique:** Correct masking logic. The include_eot_in_loss option is there but defaults to False (correct - don't train to predict eot from content).

#### `niels_gpt/cache/cli.py`
**Purpose:** CLI for building all caches.

**Builds:**
- fineweb_edu (streaming)
- wikitext (full load)
- roam (if present)
- dolly15k (SFT)
- oasst1 (SFT)

**Critique:** Works but missing primer support in CLI. Gutenberg is stub-only. No progress bars for long operations.

### 7.4 Training

#### `train/pretrain.py`
**Purpose:** Pretrain training loop.

**Data loading:**
- `PretrainSource`: Memory-mapped shards
- `PretrainMixture`: Sample from sources with probabilities

**Training loop:**
```python
for step in range(total_steps):
    lr = lr_at_step(step, ...)
    optimizer.zero_grad()
    for _ in range(accum_steps):
        x, y = mixture.get_batch(...)
        with amp_ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)
        (loss / accum_steps).backward()
    clip_grad_norm_(...)
    optimizer.step()
```

**Checkpointing:**
- latest.pt every ckpt_every
- best.pt when val improves
- Step-named checkpoints

**Critique:** Solid implementation. The parameter group logic for weight decay is correct. Loss moving average window (50) is reasonable. 

**Issue:** No gradient scaling with AMP - relies on autocast alone. Usually fine but could cause issues with fp16 on edge cases.

#### `train/sft.py`
**Purpose:** SFT training loop.

Very similar to pretrain with:
- SFT mixture instead of pretrain mixture
- ignore_index=-100 for masked loss
- Can init from pretrain checkpoint

**Critique:** Clean. The masking via y_masked with -100 is correct PyTorch pattern.

#### `train/run.py`
**Purpose:** CLI dispatcher for training phases.

**Phases:**
- pretrain: Run pretrain only
- sft: Run SFT only
- pipeline: Pretrain → SFT

**Critique:** Simple and correct. Pipeline doesn't support resuming SFT phase if pretrain completes but SFT fails - acceptable limitation.

### 7.5 Inference

#### `niels_gpt/generate.py`
**Purpose:** Text generation utilities.

**`generate_ids()`:**
- Full forward pass each token (no cache)
- Top-k/p filtering
- Repetition penalty
- Stop on eot_id

**`generate_ids_cached()`:**
- Uses KV-cache
- Same sampling options
- Optional attention tracing

**`generate_ids_greedy_full()` / `generate_ids_greedy_cached()`:**
- Greedy (argmax) variants for testing

**Critique:** Both cached and non-cached paths exist. The non-cached path is slow but useful for simple testing. The cached path follows spec correctly.

**Issue:** `generate_ids()` (non-cached) crops context to last T tokens if input exceeds T. This is correct but could lose important context. The cached version raises instead (more conservative).

#### `niels_gpt/chat_cli.py`
**Purpose:** Interactive chat interface.

**Flow:**
1. Load checkpoint
2. Build chat history with system prompt
3. Loop: input → format → generate → extract reply

**Critique:** Works correctly. Uses CPU generator for cross-device determinism. Bans role tokens during generation to prevent prompt injection.

### 7.6 KV-Cache

#### `niels_gpt/infer/kv_cache.py`
**Purpose:** KV-cache infrastructure.

**`KVCache` dataclass:**
```python
k: Tensor  # (L, B, H, T_max, D)
v: Tensor  # (L, B, H, T_max, D)
t: int     # current position
```

**`prefill()`:**
- Process full prompt
- Fill cache positions 0:t0
- Return logits

**`decode_step()`:**
- Process single token
- Append to cache at position t
- Increment t

**Critique:** Excellent implementation. Follows spec exactly. Hard cap enforcement at T_max with clear errors. Supports attention tracing.

**Invariants verified:**
- Model must be in eval mode (caller responsibility)
- Batch size consistency
- Position bounds

---

## 8. Design Decisions and Alternatives

### 8.1 Architecture Choices

| Choice | Alternative | Why This Choice |
|--------|-------------|-----------------|
| RoPE | Learned absolute, ALiBi | Better extrapolation than learned, more standard than ALiBi |
| RMSNorm | LayerNorm | Simpler, slightly faster, equally effective |
| Pre-norm | Post-norm | More stable for deep networks |
| SwiGLU | GELU, ReLU | Better perplexity at similar cost |
| Weight tying | Separate head | Fewer params, slightly better perplexity |
| Unigram tokenizer | BPE | More stable on diverse data |

### 8.2 Training Choices

| Choice | Alternative | Why This Choice |
|--------|-------------|-----------------|
| AdamW | SGD, Adam | Standard for transformers, decoupled weight decay |
| Cosine decay | Step decay, linear | Smooth, no hyperparameter for step schedule |
| Gradient clipping | None, adaptive | Simple, prevents explosion |
| AMP (fp16) | fp32 only, bf16 | Speed on MPS, fp16 more compatible than bf16 |

### 8.3 What's Missing (vs. Production)

1. **Flash Attention:** This uses naive O(T²) attention. Flash Attention is O(T) memory and faster but requires CUDA.

2. **Gradient checkpointing by layer:** Only full checkpointing supported, not selective.

3. **bf16 training:** MPS has limited bf16 support. Production uses bf16 on modern GPUs.

4. **Distributed training:** Single device only. Production uses data/tensor/pipeline parallelism.

5. **Streaming generation:** No token-by-token streaming interface. Chat CLI waits for full response.

6. **Quantization:** No int8/int4 support for inference.

### 8.4 Known Issues

1. **Primer cache not in CLI:** Must build manually.

2. **Fixed special token IDs:** If you retrain tokenizer differently, must update settings.

3. **No eval metrics beyond loss:** No perplexity, no downstream benchmarks.

4. **No checkpoint pruning:** Can accumulate many step_*.pt files.

---

## End-to-End Mental Model

1. **Data Preparation:**
   - Collect text (roam, wiki, fineweb)
   - Train tokenizer (SentencePiece unigram)
   - Build token caches (sharded .bin files)

2. **Pretraining:**
   - Sample random windows from caches
   - Train next-token prediction
   - Save best checkpoint by val loss

3. **SFT:**
   - Load pretrain checkpoint
   - Train on conversations
   - Mask loss to assistant tokens only

4. **Inference:**
   - Format chat with special tokens
   - Prefill KV-cache with prompt
   - Decode one token at a time
   - Sample with temperature/top-k/p
   - Stop on eot token

Every design choice balances speed, memory, and quality. This codebase is toy-scale but architecturally sound. The patterns transfer to production with better hardware and more data.

---

## 9. Data Loaders Deep Dive

### 9.1 Data Type Definitions (`niels_gpt/data/types.py`)

```python
@dataclass(frozen=True)
class PretrainSample:
    text: str       # Raw text
    source: str     # Dataset name
    meta: dict      # Any metadata

@dataclass(frozen=True)
class ChatMessage:
    role: Role      # "system" | "user" | "assistant"
    content: str

@dataclass(frozen=True)
class ChatSample:
    messages: list[ChatMessage]
    source: str
    meta: dict
```

### 9.2 Wikitext Loader (`niels_gpt/data/wikitext.py`)

**What it does:**
- Loads `wikitext-103-raw-v1` from HuggingFace
- Filters empty lines
- Validates no special token collisions

```python
def iter_wikitext(config="wikitext-103-raw-v1", split="train", take=None):
    ds = load_dataset("wikitext", config, split=split)
    for idx, sample in enumerate(ds):
        text = sample["text"]
        if text.strip() == "":
            continue
        assert_no_special_collision(text, ...)
        yield PretrainSample(text=text, source="wikitext", meta={"index": idx})
```

**Critique:** Simple and correct. No streaming needed since wikitext fits in memory.

### 9.3 Roam Loader (`niels_gpt/data/roam.py`)

**What it does:**
- Lists all `.md` files recursively under roam dir
- Deterministic split by file (not bytes)
- Reads with `errors="replace"` (replaces invalid UTF-8 with U+FFFD)

```python
def split_roam_paths(paths, val_frac=0.1, seed=42):
    shuffled = paths.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(paths) * val_frac))
    return shuffled[val_size:], shuffled[:val_size]
```

**Critique:** Works but doesn't weight by file size. A single large file could dominate val or train by chance.

### 9.4 Dolly Loader (`niels_gpt/data/dolly.py`)

**What it does:**
- Loads `databricks/databricks-dolly-15k`
- Maps to user/assistant format
- Optionally adds system prompt

**Mapping:**
```
instruction -> user content
context (if non-empty) -> appended to user with "\n\ncontext:\n"
response -> assistant content
```

**Critique:** Good mapping. The system prompt option exists but defaults to off (correct for this dataset).

### 9.5 OASST1 Loader (`niels_gpt/data/oasst1.py`)

**What it does:**
- Loads `OpenAssistant/oasst1` message trees
- Reconstructs all root→leaf conversation paths
- Filters to English only by default
- Caps at 32 messages per thread

**Reconstruction:**
```python
def _reconstruct_threads(records, english_only=True, max_messages=32):
    # Build parent→children lookup
    # DFS from each root to all leaves
    # Emit complete paths as threads
```

**Role mapping:**
- `prompter` → `user`
- `assistant` → `assistant`

**Sanitization:**
- Skips empty messages
- Skips threads with consecutive same-role messages
- Strips whitespace

**Critique:** Sophisticated reconstruction. The consecutive same-role skip is conservative but correct (can't safely merge without losing context).

### 9.6 Batching (`niels_gpt/batching.py`)

**For byte-level (legacy):**
```python
def get_batch(sources, p, B, T, device, generator):
    # Sample which source for each batch item
    source_indices = torch.multinomial(probs, B, replacement=True, generator=generator)
    
    for idx in source_indices:
        source = sources[source_names[idx]]
        # Random window of T+1 bytes
        start = torch.randint(0, len(source) - T, (1,), generator=generator)
        chunk = source[start:start+T+1]
        x = chunk[:-1]
        y = chunk[1:]
```

**Validation:**
- Keys must match exactly between sources and p
- Probabilities must sum to 1.0
- Each source must have at least T+1 bytes

**Critique:** The byte-level batching is for legacy code. Main training uses token-level shards via `PretrainSource`.

---

## 10. Complete Config Schema

### 10.1 Settings Hierarchy

```
Settings (root)
├── tokenizer: TokenizerSettings
│   ├── model_path: str
│   ├── vocab_size: int
│   ├── special_tokens: SpecialTokens
│   └── expected_special_token_ids: dict
├── data: DataSettings
│   ├── caches: CacheSettings
│   ├── mix_pretrain: dict[str, float]
│   ├── mix_sft: dict[str, float]
│   ├── val_pretrain_source: str
│   └── val_sft_source: str
├── model: ModelSettings
│   ├── V, T, C, L, H, d_ff, dropout, rope_theta
│   ├── norm_type: "rmsnorm" | "layernorm"
│   └── mlp_type: "swiglu" | "gelu"
├── training: TrainingSettings
│   ├── pretrain: TrainPhaseSettings
│   └── sft: TrainPhaseSettings
├── sft_format: SFTFormattingSettings
│   ├── assistant_only_loss: bool
│   ├── include_eot_in_loss: bool
│   └── ban_role_tokens_during_generation: bool
├── generation: GenerationSettings
│   ├── max_new_tokens, temperature, top_k, top_p
│   └── repetition_penalty
└── benchmark: BenchmarkSettings
```

### 10.2 Override Examples

**Small model for fast iteration:**
```json
{
  "model": {"T": 256, "C": 256, "L": 4, "H": 4},
  "training": {
    "pretrain": {"total_steps": 1000, "micro_B": 32}
  }
}
```

**Large model with AMP:**
```json
{
  "model": {"T": 1024, "C": 768, "L": 16, "H": 12},
  "training": {
    "pretrain": {
      "micro_B": 8,
      "accum_steps": 8,
      "amp": true,
      "activation_checkpointing": true
    }
  }
}
```

**Custom data mix:**
```json
{
  "data": {
    "mix_pretrain": {"wikitext": 0.5, "roam": 0.5}
  }
}
```

---

## 11. Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| `tokenizer not found` | Missing artifacts | Run `python scripts/train_tokenizer.py` |
| `cache file not found` | Missing cache build | Run `python -m niels_gpt.cache.cli build-all` |
| `source too short` | Data smaller than T+1 | Reduce T or add more data |
| `tokenizer hash mismatch` | Retrained tokenizer | Rebuild caches |
| `special token collision` | Raw data contains sentinel | Edit data or change UUID |
| `OOM on MPS` | Batch/context too large | Reduce micro_B, reduce T, enable activation_checkpointing |
| `NaN loss` | Training instability | Reduce lr, disable AMP, check data |
| `model config mismatch` | Wrong checkpoint | Use checkpoint matching current config |

---

## 12. Summary Critique

### What's Good

1. **Clean architecture:** Standard transformer with modern choices (RoPE, RMSNorm, SwiGLU)
2. **Good safety:** Special token collision detection, tokenizer validation
3. **Reproducibility:** Deterministic seeds, config checksums, run tracking
4. **KV-cache:** Proper implementation with hard cap enforcement
5. **Settings system:** Pydantic validation, deep merge, resolved output

### What Could Be Better

1. **No Flash Attention:** O(T²) limits context length
2. **Fixed special token IDs:** Brittle if retrained
3. **No distributed training:** Single device only
4. **No streaming generation:** Full response before output
5. **Primer not in CLI:** Manual cache build required
6. **No CUDA path:** Auto-detect only covers MPS/CPU

### Production Readiness

This codebase is **educational/experimental** quality:
- ✅ Architecturally sound
- ✅ Good for learning
- ⚠️ Not optimized for speed
- ⚠️ Limited to ~100M params practical
- ❌ Missing production features (quantization, distributed, etc.)

For production, use this as a learning reference and build with vLLM, TensorRT-LLM, or similar.
