pr-05b spec: causal self-attention + causal mask + rope integration + tests

goal

implement a correct, testable causal self-attention module (from scratch) that:
	•	consumes x: (B,T,C) and returns y: (B,T,C)
	•	applies rope to q and k using your existing rope_cache/apply_rope
	•	enforces a causal mask (no attention to future tokens)
	•	supports return_attn=True for tests
	•	includes pytest tests that fail loudly on common bugs

allowed changes

may create/modify only:
	•	niels_gpt/model/blocks.py
	•	tests/test_attention_mask.py
	•	optionally niels_gpt/model/__init__.py (only if needed for imports)

must not modify:
	•	niels_gpt/model/rope.py (already implemented)
	•	anything under niels_gpt/config.py, niels_gpt/device.py, data code, etc.

dependencies / imports
	•	config: from niels_gpt.config import ModelConfig
	•	rope module must be imported as a module so pytest monkeypatch works:
	•	import niels_gpt.model.rope as rope
	•	and call rope.apply_rope(...) (do not from rope import apply_rope)

public API to implement

in niels_gpt/model/blocks.py, implement:

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C) float32
        returns:
          - y: (B, T, C) float32  if return_attn=False
          - (y, attn): where attn is (B, H, T, T) float32  if return_attn=True
        """

notes
	•	cfg.T, cfg.C, cfg.H, cfg.dropout, cfg.rope_theta are authoritative.
	•	D = cfg.C // cfg.H must be an integer and must be even (rope requirement).
	•	v0 dtype is float32, but don’t hardcode float32 in a way that breaks .to(device).

exact architecture constraints

qkv projection
	•	single projection: nn.Linear(cfg.C, 3 * cfg.C, bias=False)
	•	split into q/k/v along last dim
	•	reshape to (B, H, T, D) (canonical)

rope integration
	•	build sin/cos buffers once in __init__ using your rope cache:

sin, cos = rope.rope_cache(cfg.T, D, theta=cfg.rope_theta, device="cpu", dtype=torch.float32)
self.register_buffer("rope_sin", sin, persistent=False)
self.register_buffer("rope_cos", cos, persistent=False)

	•	in forward, after q/k reshaped to (B,H,T,D):
	•	q, k = rope.apply_rope(q, k, self.rope_sin, self.rope_cos)

causal mask
	•	create a lower-triangular boolean mask in __init__, register as buffer:

mask = torch.tril(torch.ones(cfg.T, cfg.T, dtype=torch.bool))
mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
self.register_buffer("mask", mask, persistent=False)

	•	apply mask before softmax, using torch.finfo(scores.dtype).min for masked values:

scores = scores.masked_fill(~self.mask[:, :, :T, :T], torch.finfo(scores.dtype).min)

attention math

given q, k, v: (B,H,T,D):
	•	scores = (q @ k.transpose(-2, -1)) * (1 / sqrt(D))  → (B,H,T,T)
	•	mask
	•	attn = softmax(scores, dim=-1) → (B,H,T,T)
	•	attn = attn_dropout(attn) in train mode
	•	out = attn @ v → (B,H,T,D)
	•	merge heads: transpose + reshape → (B,T,C)
	•	output projection: nn.Linear(C, C, bias=False)
	•	residual dropout after projection: resid_dropout

dropout
	•	self.attn_dropout = nn.Dropout(cfg.dropout) applied to attn probs (post-softmax)
	•	self.resid_dropout = nn.Dropout(cfg.dropout) applied to output projection result
	•	tests must use module.eval() to disable dropout.

no initialization in this PR

do not add global init logic here. pr-05c will handle init. keep this PR purely functional + tested.

acceptance tests

create tests/test_attention_mask.py with:

test 1: shape/dtype/device + rope-called
	•	create small cfg: ModelConfig(T=8, C=32, H=4, d_ff=128, dropout=0.1) (D=8, even)
	•	create module: attn = CausalSelfAttention(cfg).to(device)
	•	input: x = torch.randn(2, cfg.T, cfg.C, device=device)
	•	monkeypatch niels_gpt.model.rope.apply_rope to a wrapper that:
	•	asserts q.shape == (B,H,T,D) and same for k
	•	increments a counter
	•	calls the real function (or returns q,k unchanged if you want pure call-check)
	•	run y = attn(x)
	•	assert:
	•	output shape (2, T, C)
	•	y.dtype == x.dtype
	•	y.device == x.device
	•	rope apply wrapper called exactly once per forward

test 2: causal mask has zero mass above diagonal
	•	use same cfg/module but call attn.eval()
	•	run (y, a) = attn(x, return_attn=True)
	•	assert a.shape == (B, H, T, T)
	•	compute upper triangle mask (above diagonal): torch.triu(torch.ones(T,T,dtype=bool), diagonal=1)
	•	assert a[..., upper].max() is very small (e.g. <= 1e-6)
	•	also assert each row sums to ~1: a.sum(dim=-1) close to ones (tolerance e.g. 1e-5)

test runner

must pass: pytest -q

definition of done
	•	pytest -q passes
	•	implementation obeys all constraints above
	•	no changes outside allowlist
