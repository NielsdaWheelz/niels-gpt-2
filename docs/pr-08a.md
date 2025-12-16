# PR-08a: traceable forward for attention inspection (niels-gpt repo)

## goal
Expose attention weights for a selected transformer layer in a way that:
- keeps existing training/generation code unchanged
- allows streaming a small “attention row” per generated token
- optionally allows requesting the full attention matrix on-demand

## non-goals
- changing GPT.forward() signature/behavior
- returning traces for all layers by default
- adding new dependencies
- changing training or generation modules

## repo facts (existing)
- attention module: `niels_gpt/model/blocks.py`
- `CausalSelfAttention.forward(x, return_attn: bool=False)`:
  - returns output `(B,T,C)` if return_attn False
  - returns `(output, attn_probs)` if True, where `attn_probs` is `(B,H,T,T)` **pre-dropout**
- `Block.forward(x, return_attn: bool=False)` mirrors that behavior and returns `(x, attn_probs)` if return_attn True

## required new public API

### in `niels_gpt/model/gpt.py`
Add a new method to `GPT`:

```py
from typing import TypedDict, Optional
import torch

class AttnTrace(TypedDict):
    layer: int
    attn_row: torch.Tensor           # (B, H, t)
    attn_full: Optional[torch.Tensor]  # (B, H, t, t) if requested else None

class GPT(nn.Module):
    def forward_with_attn_trace(
        self,
        x: torch.LongTensor,
        *,
        trace_layer: int,
        return_full_attn: bool = False,
    ) -> tuple[torch.Tensor, AttnTrace]:
        """
        x: (B, t) int64 token ids, where t <= cfg.T
        trace_layer: which block to trace, 0 <= trace_layer < cfg.L
        return_full_attn:
          - False: return only attn_row
          - True: also return attn_full

        returns:
          logits: (B, t, V) float tensor (same as forward)
          trace:
            - layer: trace_layer
            - attn_row: attention probs for the last token (query pos = t-1):
                attn_probs[:, :, t-1, :t]  -> shape (B, H, t)
            - attn_full: full matrix (B, H, t, t) if requested else None
        """

invariants
	•	GPT.forward(x) remains unchanged in output and semantics.
	•	forward_with_attn_trace must produce logits numerically identical to forward when model.eval() and the same input are used.
	•	trace_layer must be validated:
	•	raise ValueError if out of bounds.
	•	returned attention probabilities must be the pre-dropout probabilities already produced by CausalSelfAttention (i.e., attn_probs).
	•	shapes must match exactly:
	•	attn_row: (B, H, t)
	•	attn_full: None or (B, H, t, t)
	•	attention rows should be proper distributions:
	•	for each (B,H), attn_row.sum(dim=-1) ~= 1 (within tolerance)

implementation constraints (keep it simple)
	•	do NOT change blocks.py interfaces unless strictly required.
	•	implement tracing by running the normal forward loop over blocks, but:
	•	for the selected trace_layer, call block(x, return_attn=True) and capture attn_probs.
	•	for all other layers, call block(x) as usual.
	•	ensure masking still applies (already handled in blocks).
	•	do not allocate or return full (t,t) matrices unless return_full_attn=True.

tests (MUST ADD)

ADD: tests/test_attn_trace.py

Tests must run quickly on CPU and not require trained weights.
	1.	test_forward_unchanged_eval_allclose

	•	build a small config for speed (override ModelConfig values; keep V=256)
	•	instantiate GPT, set eval()
	•	generate random token ids x with shape (B=2, t=17) and values 0..255
	•	compute logits_a = model(x)
	•	compute logits_b, trace = model.forward_with_attn_trace(x, trace_layer=0, return_full_attn=False)
	•	assert logits shapes equal and torch.allclose(logits_a, logits_b, atol=1e-6, rtol=0)

	2.	test_trace_shapes_and_sums

	•	same model/input, trace_layer=1 (ensure cfg.L >= 2 in test config)
	•	check:
	•	trace["layer"] == 1
	•	attn_row.shape == (B, H, t)
	•	attn_full is None when return_full_attn False
	•	attn_row.sum(-1) is allclose to ones with tolerance (e.g. atol=1e-4)

	3.	test_full_attn_optional

	•	call with return_full_attn=True
	•	verify:
	•	attn_full.shape == (B, H, t, t)
	•	attn_full.sum(-1) allclose to ones (atol=1e-4)
	•	attn_row matches attn_full[:, :, -1, :] (allclose)

	4.	test_trace_layer_bounds

	•	out of range trace_layer raises ValueError (negative, >= cfg.L)

allowed file changes

MODIFY:
	•	niels_gpt/model/gpt.py

ADD:
	•	tests/test_attn_trace.py

DO NOT MODIFY:
	•	training loop
	•	generation utilities
	•	tokenizer/chat formatting
	•	checkpointing
	•	blocks.py (unless absolutely necessary; prefer not to)

done when
	•	pytest passes
	•	forward_with_attn_trace works on CPU and MPS (no device assumptions in code)
