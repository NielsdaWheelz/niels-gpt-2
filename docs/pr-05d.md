pr-05d-spec

goal

add the GPT wrapper model that composes existing building blocks into an end-to-end byte-level causal LM, with correct initialization and weight tying, plus tests that catch wiring mistakes.

scope

must create
	•	niels_gpt/model/gpt.py
	•	tests/test_gpt.py (or tests/test_gpt_forward.py — pick one; prefer test_gpt.py)

may modify (only if needed for imports/exports)
	•	niels_gpt/model/__init__.py (export GPT)

must not modify
	•	niels_gpt/model/blocks.py
	•	niels_gpt/model/rope.py
	•	any data-loading code

dependencies (already implemented)
	•	niels_gpt.config.ModelConfig (frozen dataclass)
	•	niels_gpt.model.blocks.Block:
	•	__init__(self, cfg: ModelConfig)
	•	forward(self, x: torch.Tensor, *, return_attn: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
	•	niels_gpt.model.blocks.init_weights(module: nn.Module) -> None

public API

class GPT
	•	location: niels_gpt/model/gpt.py
	•	signature:

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor: ...

forward contract
	•	input:
	•	x: shape (B, T_cur) dtype torch.int64
	•	token ids in [0..cfg.V-1]
	•	output:
	•	logits: shape (B, T_cur, cfg.V) dtype torch.float32
	•	invariants:
	•	allow variable length T_cur <= cfg.T
	•	raise AssertionError (or ValueError) if T_cur > cfg.T

architecture (v0 hard constraints)

build a decoder-only transformer with RoPE positions handled inside blocks.

modules
	•	token embedding: tok_emb = nn.Embedding(cfg.V, cfg.C)
	•	dropout at embeddings: drop = nn.Dropout(cfg.dropout)
	•	transformer blocks: blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.L)])
	•	final layernorm: ln_f = nn.LayerNorm(cfg.C, eps=1e-5)
	•	lm head: lm_head = nn.Linear(cfg.C, cfg.V, bias=False)

weight tying (hard)
	•	tie weights by object identity:
	•	self.lm_head.weight = self.tok_emb.weight
	•	must be asserted in tests.

initialization (hard)
	•	in GPT.__init__, after module creation and weight tying:
	•	call self.apply(init_weights)
	•	rationale: modules do not self-init; tests currently apply init manually, but GPT must own it.

no learned position embeddings (hard)
	•	do not add pos_emb or any absolute positional embedding. RoPE is already applied inside attention.

forward pass behavior (hard)

given x with shape (B, T_cur):
	1.	h = tok_emb(x) → (B, T_cur, C)
	2.	h = drop(h) (dropout respects .train() / .eval())
	3.	for each block in blocks: h = block(h) (ignore attention returns; do not pass return_attn)
	4.	h = ln_f(h)
	5.	logits = lm_head(h) → (B, T_cur, V)
	6.	return logits

notes:
	•	blocks are responsible for causal masking and RoPE.
	•	GPT.forward must return logits only (never a tuple).

tests (required)

add tests/test_gpt.py with at least these tests:

test 1: shape/dtype/device
	•	create cfg = ModelConfig()
	•	device = niels_gpt.device.get_device()
	•	model = GPT(cfg).to(device)
	•	x = torch.randint(0, cfg.V, (2, cfg.T), dtype=torch.long, device=device)
	•	logits = model(x)
	•	assert:
	•	logits.shape == (2, cfg.T, cfg.V)
	•	logits.dtype == torch.float32
	•	logits.device.type == device

test 2: variable length
	•	x = torch.randint(0, cfg.V, (2, 17), dtype=torch.long, device=device)
	•	logits = model(x)
	•	assert shape (2, 17, cfg.V)

test 3: reject too-long
	•	x = torch.randint(0, cfg.V, (1, cfg.T + 1), dtype=torch.long, device=device)
	•	assert it raises (AssertionError or ValueError)

test 4: weight tying
	•	assert model.lm_head.weight is model.tok_emb.weight

test 5: backward smoke (finite grads)
	•	x as in test 1 (or smaller T to speed)
	•	y = torch.randint(0, cfg.V, (B, T_cur), dtype=torch.long, device=device)
	•	logits = model(x)
	•	loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
	•	loss.backward()
	•	assert a representative grad is finite (e.g. model.tok_emb.weight.grad exists and torch.isfinite(...).all())

constraints:
	•	tests must run on cpu and mps
	•	keep sizes small enough that cpu tests are fast (you can use T_cur=32 in backward smoke)

non-goals
	•	generation code
	•	training loop
	•	optimizer or scheduler code
	•	any changes to existing blocks/rope implementations

acceptance criteria
	•	pytest -q passes
	•	GPT(cfg) produces correct logits shape on cpu and mps
	•	weight tying and init are correctly applied
