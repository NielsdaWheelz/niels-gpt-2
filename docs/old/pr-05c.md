pr-05c spec

goal
add MLP, Block (pre-norm transformer block wiring), and init_weights to make the model trainable + testable. include tight unit tests for shape, grads, and dropout semantics.

non-goals
	•	no changes to rope (pr-05a)
	•	no changes to attention internals (pr-05b) beyond importing/using CausalSelfAttention
	•	no GPT wrapper yet (pr-05d)
	•	no training loop, no dataset code

allowed changes
	•	may edit/create only:
	•	niels_gpt/model/blocks.py
	•	tests/test_block.py (name can differ if you have a convention; keep it under tests/)
	•	must not touch anything else.

required public surface area

MLP
	•	location: niels_gpt/model/blocks.py
	•	signature:

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

contract
	•	input x: (B, T, C) float32
	•	output: (B, T, C) float32
	•	architecture: Linear(C, d_ff, bias=True) -> GELU -> Linear(d_ff, C, bias=True)
	•	d_ff = cfg.d_ff

Block
	•	location: niels_gpt/model/blocks.py
	•	signature:

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig): ...

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...

contract
	•	uses pre-norm residual wiring:

x = x + dropout(attn(ln1(x)))
x = x + dropout(mlp(ln2(x)))

	•	ln1, ln2: nn.LayerNorm(cfg.C, eps=1e-5)
	•	dropout: nn.Dropout(cfg.dropout) reused for both residual branches
	•	must call your existing attention class:

from niels_gpt.model.blocks import CausalSelfAttention  # or local name if already in file
# instantiate: self.attn = CausalSelfAttention(cfg)
# forward must pass return_attn through

return semantics
	•	if return_attn=False: return x
	•	if return_attn=True: return (x, attn_probs) where attn_probs is exactly the second output from CausalSelfAttention(..., return_attn=True)

init_weights
	•	location: niels_gpt/model/blocks.py
	•	signature:

def init_weights(module: nn.Module) -> None:
    ...

contract
	•	implements explicit gpt-ish init:
	•	for nn.Linear and nn.Embedding: weight.normal_(mean=0.0, std=0.02)
	•	for nn.Linear bias (if present): zero
	•	for nn.LayerNorm: weight ones, bias zeros
	•	must be safe to call via model.apply(init_weights) on arbitrary module trees.

implementation constraints
	•	do not introduce new dependencies
	•	do not use scaled_dot_product_attention
	•	do not change the signature of CausalSelfAttention (already implemented)
	•	keep everything float32-friendly

⸻

tests (must be added in this pr)

test 1: mlp shape + grads are finite
	•	instantiate cfg = ModelConfig() (or a copy with dropout irrelevant)
	•	mlp = MLP(cfg); mlp.apply(init_weights)
	•	x = torch.randn(2, cfg.T, cfg.C, dtype=torch.float32)
	•	out = mlp(x) shape (2, cfg.T, cfg.C)
	•	loss = out.pow(2).mean(); loss.backward()
	•	assert at least one parameter has grad is not None
	•	assert no nan/inf in grads

test 2: block forward/backward smoke
	•	block = Block(cfg); block.apply(init_weights)
	•	x = torch.randn(2, cfg.T, cfg.C)
	•	run in eval mode once, assert output shape correct
	•	run with return_attn=True, assert:
	•	returns (y, attn)
	•	y shape (2, cfg.T, cfg.C)
	•	attn is a tensor and its last two dims are (cfg.T, cfg.T) (don’t over-specify beyond that)
	•	backward: y.mean().backward() and assert finite grads

test 3: dropout semantics

we want non-flaky checks:
	•	create cfg_drop = replace(ModelConfig(), dropout=0.5) (or construct a new ModelConfig if replace isn’t available; since it’s frozen, use dataclasses.replace)
	•	block = Block(cfg_drop); block.apply(init_weights)
	•	x = torch.randn(2, cfg_drop.T, cfg_drop.C)

assertions:
	1.	block.eval() → two forwards identical:
	•	y1 = block(x)
	•	y2 = block(x)
	•	torch.testing.assert_close(y1, y2)
	2.	block.train() → two forwards differ without reseeding:
	•	torch.manual_seed(123); y1 = block(x)
	•	y2 = block(x)
	•	assert not torch.allclose(y1, y2)

(this should be stable with dropout=0.5 and big tensors.)

⸻

acceptance criteria
	•	pytest -q passes
	•	no files outside the allowlist changed
	•	MLP, Block, init_weights exist exactly as specified
	•	Block(..., return_attn=True) returns (x, attn_probs) and passes through attention’s attn_probs
