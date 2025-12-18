pr roadmap: “make training sane + primer-as-sft + fast chat”

pr-01 — canonical source keys + fail-fast validation (break compat)

goal
	•	break backwards compatibility completely: enforce wikitext (no wiki)
	•	validate cache presence before training starts
	•	print resolved sources + sizes so you can’t accidentally train on the wrong mix

changes
	•	niels_gpt/settings.py
	•	restrict data.mix_pretrain keys to an enum set (must include wikitext, may include fineweb_edu, gutenberg, roam)
	•	restrict data.mix_sft keys to {dolly15k, oasst1, primer} (primer added later but reserve now)
	•	set defaults so they are valid given existing caches
	•	train/pretrain.py (or wherever settings are materialized)
	•	add validate_pretrain_caches(cache_dir, mix_keys) that checks:
	•	{cache_dir}/{src}/meta.json exists
	•	{cache_dir}/{src}/{train,val}/ exists
	•	log: for each src, print num_tokens, num_shards, and split sizes from meta
	•	train/sft.py
	•	same cache validation for sft sources

acceptance tests
	•	unit: config with mix_pretrain={"wiki":1.0} raises ValueError at settings load time
	•	unit: missing cache dir produces a clean FileNotFoundError listing exact missing paths

non-goals
	•	no alias map, no auto rename, no “helpful” fallback

⸻

pr-02 — sft eval defaults + report both curves (stop lying to yourself)

goal
	•	make sft-val actually evaluate sft data by default
	•	always report:
	•	val_pretrain_loss (wikitext)
	•	val_sft_loss (sft val)
so you can see when chat improves while LM loss worsens (or vice versa)

changes
	•	niels_gpt/settings.py
	•	change val_sft_source default from "wikitext" → "sft"
	•	make the choice explicit enum: {"sft","wikitext"} if you insist on the old behavior
	•	train/sft.py
	•	simplify selection logic:
	•	if val_source_choice=="sft": load sft val mixture and call evaluate_sft
	•	else: call evaluate_pretrain (but label it as pretrain-val)
	•	during eval, compute both if you want: pretrain val can be optional, but recommended to always compute at least wikitext val occasionally
	•	train/logging.py (or wherever you print metrics)
	•	ensure metrics keys are stable and explicit (val_pretrain_loss, val_sft_loss)

acceptance tests
	•	unit: default settings produce val_source_choice=="sft"
	•	unit: sft runner calls evaluate_sft with ignore_index=-100

⸻

pr-03 — primer becomes an sft dataset (format + tokenization + caching)

goal
	•	treat primer as sft, not pretrain
	•	define a primer sft format that uses your special tokens and trains “assistant-only” loss
	•	build token caches to data/cache/sft/primer/... with the same sharded .bin + meta.json structure as other sft sources

decisions locked in this PR
	•	primer lives under data/cache/sft/primer/{train,val}/...
	•	primer input file: data/primer.jsonl (or keep primer.txt but then you must define a deterministic parser)
	•	template: <|sys|>...<|eot|><|usr|>...<|eot|><|asst|>...<|eot|>
	•	labels masked to -100 except assistant tokens including <|eot|>

changes
	•	niels_gpt/data/primer_sft.py (new)
	•	load jsonl → list of message dicts
	•	apply chat template using the tokenizer’s special ids
	•	produce:
	•	input_ids: list[int]
	•	labels: list[int] with masking
	•	niels_gpt/cache/cli.py
	•	add command: build-sft-primer (or extend build-all; see next PR)
	•	write sharded token cache in the same pattern as dolly/oasst1:
	•	shard_00000.bin
	•	meta.json containing num_tokens, shard sizes, tokenizer hash, etc.
	•	train/sft.py
	•	allow primer as an sft source name (same loader path as others)
	•	niels_gpt/settings.py
	•	include primer in allowed sft sources
	•	remove primer from default mix_pretrain if it exists there today

acceptance tests
	•	unit: primer example produces correct special tokens and masking
	•	unit: labels[i] == -100 for sys/usr spans, equals token id for asst span
	•	cache build produces expected directory tree and meta

non-goals
	•	no packing across examples, no curriculum, no splitting “structure vs voice” (you deferred it)

⸻

pr-04 — build-all becomes actually “build all required defaults”

goal
	•	remove the “manual step” footgun
	•	if defaults require primer, build-all builds primer; otherwise build-all errors loudly when mix references missing caches

changes
	•	niels_gpt/cache/cli.py
	•	update build-all to include primer sft cache build
	•	verify: build-all ends by validating caches for default configs (pretrain + sft)
	•	docs cleanup:
	•	pick one authoritative doc (README or QUIKSTART). the other should link to it.

acceptance tests
	•	unit/integration-ish: run build-all in a temp dir and assert the primer cache dirs exist
	•	unit: if primer is referenced in sft mix and cache missing, training fails before step 0 with a clear message

⸻

pr-05 — chat cli uses kv-cache by default

goal
	•	the demo must feel fast without requiring the user to know about your internal generate functions

changes
	•	niels_gpt/chat_cli.py
	•	switch from generate_text (uncached) to cached path
	•	add --no-kv-cache flag to revert
	•	ensure eot stopping is token-based only (you already want no substring hacks)
	•	niels_gpt/generate.py
	•	if generate_text is a wrapper, make it call cached by default (optional) or add generate_text_cached

acceptance tests
	•	unit: CLI chooses cached path unless flag set (mock the function call)
	•	unit: stops on eot token id

⸻

pr-06 — remove triple dropout (optional but recommended soon)

goal
	•	stop over-regularizing your tiny model

changes
	•	niels_gpt/model/blocks.py
	•	remove Block.dropout wrapping of attn/mlp outputs (or set it to identity)
	•	keep attention’s attn_dropout + resid_dropout, and mlp’s dropout (if any) in one canonical place

acceptance tests
	•	unit: forward shape unchanged, training still runs
	•	(optional) snapshot test: ensure no dropout module count explosion

⸻

ordering + why
	•	pr-01 first because everything else depends on keys being sane.
	•	pr-02 second because otherwise you can’t trust your training signals.
	•	pr-03/04 next because primer-as-sft is a pipeline change + tooling.
	•	pr-05 after because it’s pure product wiring and shouldn’t block training.
	•	pr-06 last; it’s quality polish.

⸻

primer sft input format: jsonl.

primer in mix_sft by default. 
