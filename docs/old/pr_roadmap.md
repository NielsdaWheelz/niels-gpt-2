pr roadmap (v0)

pr-01: repo skeleton + config + device/seed utilities

goal: consistent config + deterministic-ish runs.
scope:
	•	config.py: ModelConfig, TrainConfig dataclasses (or pydantic if you insist, but dataclasses is simpler)
	•	device.py: get_device() (mps if available else cpu)
	•	rng.py: set_seed(42) (torch + python random; include numpy only if you actually import it)
	•	paths.py: standard paths: .roam-data/, data/primer.txt, checkpoint dir
tests:
	•	test_seed_determinism.py: same seed → same torch.randint sequence (with explicit generator)
done when: python -c "import device; print(device.get_device())" works.

pr-02: tokenizer + chat formatting/parsing

goal: lock text↔bytes and transcript rules.
scope:
	•	tokenizer.py: utf-8 byte encode/decode
	•	chat_format.py: format_chat, extract_assistant_reply
tests:
	•	tokenizer roundtrip on ascii
	•	chat formatting ends with exactly "assistant: " (space, no newline)
	•	parsing stops only on \n(system|user|assistant):  tags
done when: unit tests pass.

pr-03: datasets + splits

goal: load wikitext, roam md, primer; produce held-out splits.
scope:
	•	data/wikitext.py: load_wikitext() via datasets
	•	data/roam.py: list paths under .roam-data/, load texts, split by file (seeded)
	•	data/primer.py: load primer, split by <dialogue> blocks (seeded)
tests:
	•	roam split is deterministic given seed
	•	primer split doesn’t mix blocks
	•	wikitext loader returns train/val/test lists of strings
done when: python -c "from data.wikitext import load_wikitext; print(len(load_wikitext()['train']))" works.

pr-04: byte streams + batch sampler

goal: build bytes streams and sample training batches with p_train.
scope:
	•	streams.py: build wiki_train/wiki_val, roam_train/roam_val, primer_train/primer_val as bytes with "\n\n" separators
	•	batching.py: get_batch(sources, p, B, T, device, generator)
	•	define probs:
	•	p_train = {"wiki": 0.784, "roam": 0.196, "primer": 0.020}
tests:
	•	y[:, :-1] == x[:, 1:]
	•	sampled ids are 0..255
	•	determinism with provided torch.Generator
done when: you can print one batch shape on mps.

pr-05: model core (including rope) + forward pass

goal: implement the transformer and guarantee shape correctness.
scope:
	•	model/gpt.py: GPT(cfg) + forward(x)->logits
	•	model/blocks.py: pre-norm block, attention, mlp
	•	model/rope.py: rope_cache(T,D,theta) + apply_rope(q,k,sin,cos)
	•	causal mask
	•	optional weight tying (do it; it’s a one-liner and reduces params)
tests:
	•	forward output shape (B,T,256)
	•	rope preserves shape/dtype/device and requires even D
	•	causal mask test (upper triangle attention weights ~0) — easiest by exposing attention probs in a debug mode or testing masked logits before softmax
done when: a single forward+loss step runs.

pr-06: training loop + lr schedule + checkpointing + eval

goal: run smoke train (1000 steps) and save progress safely.
scope:
	•	train.py: loop, loss, backward, clip, step
	•	lr_schedule.py: warmup(200)+cosine to 3e-5
	•	checkpoint.py: save/load (model, optim, step, configs)
	•	eval.py: compute avg loss on val_wiki (and optionally roam/chat)
	•	config files:
	•	configs/smoke.json: steps=1000
	•	configs/train.json: steps=20000
tests:
	•	lr schedule endpoints: step0 ~0, step=warmup hits base_lr, last step ~min_lr
	•	checkpoint roundtrip loads identical model weights (at least for a small model)
done when: smoke run completes, prints loss curve, writes checkpoints, saves best.

pr-07: generation + cli chat

goal: “callable” chatbot locally.
scope:
	•	generate.py: temperature + top-k sampling, crop to last T
	•	chat_cli.py: keeps a message list, formats with format_chat, calls generate, extracts reply
tests:
	•	generate returns non-empty string
	•	deterministic generation when you fix seed + sampling off (temperature=0 / greedy)
done when: python chat_cli.py --ckpt ... feels like a chatbot.

pr-08: minimal web chat (optional)

goal: hostable endpoint + tiny web ui.
scope:
	•	minimal server (fastapi is fine) with /chat
	•	simple static html/js page
tests: basic request/response.
done when: you can chat in a browser.
