pr roadmap: v1 “best possible on m4/16gb”

pr-01: tokenizer v1 (sentencepiece unigram + special tokens)

goal: replace byte tokenizer with subword tokenizer; make stop condition token-native.

decisions locked in this PR
	•	sentencepiece unigram
	•	vocab size 16k
	•	special tokens: <|sys|> <|usr|> <|asst|> <|eot|>
	•	tokenizer trained on pretrain + sft text

deliverables
	•	tokenizer_sp.py
	•	train_tokenizer(corpus_iter, vocab_size, special_tokens) -> tokenizer_path
	•	encode(text)->ids, decode(ids)->text
	•	special_token_ids() returns ids for the 4 tokens
	•	chat_template.py
	•	format_chat_tokens(messages)->list[int] using <|sys|>...<|eot|> etc.

tests
	•	special tokens are single ids (encode returns one token for each)
	•	stop token id exists and round-trips
	•	format_chat_tokens always ends with <|asst|> (for prompting) or <|eot|> (for completed assistant turns)

⸻

pr-02: dataset loaders v1 (fineweb-edu, dolly, oasst1, gutenberg, wikitext)

goal: standardize loading into an iterator of raw text or chat examples.

deliverables
	•	data/fineweb_edu.py: streaming loader (load_dataset(..., streaming=True)) and a “take N examples” mode  ￼
	•	data/dolly.py: load databricks/databricks-dolly-15k (instruction/response fields)  ￼
	•	data/oasst1.py: load OpenAssistant/oasst1 and reconstruct conversations (tree → linear threads)  ￼
	•	data/gutenberg.py: loader stub (even if you start with a minimal curated mirror later)
	•	data/wikitext.py: keep for smoke/eval; not primary pretrain anymore

tests
	•	each loader yields non-empty samples
	•	oasst1 reconstruction produces alternating user/assistant turns (where possible)

⸻

pr-03: tokenization + on-disk token cache (the workhorse)

goal: convert raw samples into token id shards once; never re-tokenize during training.

deliverables
	•	cache/build_cache.py
	•	builds sharded .bin (uint16 or uint32) + .idx offsets + meta.json
	•	supports:
	•	pretrain_text streams (fineweb-edu, roam, wikitext, gutenberg)
	•	sft_chat streams (dolly, oasst1, your primer if you keep it)
	•	cache/dataset.py
	•	memory-map shards, sample random windows (pretrain)
	•	sample whole chat transcripts (sft), with packing/truncation to T

tests
	•	cache build is deterministic given seed + fixed “take N”
	•	mmap dataset returns correct shapes and valid token ranges

⸻

pr-04: training data format v1 (token-native chat, eot stop, packing rules)

goal: eliminate substring hacks everywhere.

deliverables
	•	format/sft.py
	•	converts each instruction/dialogue into:
	•	<|sys|>...<|eot|><|usr|>...<|eot|><|asst|>...<|eot|>
	•	defines masking rule for loss:
	•	only compute loss on assistant tokens (recommended for sft), or
	•	compute on all tokens (simpler, usually worse alignment)
	•	format/pretrain.py
	•	document separator token strategy (either newline text or a dedicated <|eot|>-like boundary token if you add one later)

tests
	•	every sft example contains at least one <|asst|> ... <|eot|> span
	•	loss-mask correctness (tokens before assistant span get ignore_index)

⸻

pr-05: model v1 architecture (rmsnorm + swiglu + rope kept)

goal: upgrade backbone while keeping your existing tracing hooks viable.

deliverables
	•	model/norm.py: RMSNorm
	•	model/mlp.py: SwiGLU MLP (with chosen hidden size rule)
	•	model/gpt_v1.py: T=1024, C=512, L=8, H=8, D=64, tied embeddings
	•	keep rope as-is, but ensure it works with new T

tests
	•	forward shape (B,T,V)
	•	training step runs (forward+loss+backward) without nan on cpu
	•	rope cache correctness for T=1024

⸻

pr-06: training v1 (two-phase runner: pretrain → sft)

goal: make “phase A then phase B” a first-class CLI, not a notebook ritual.

deliverables
	•	train/pretrain.py
	•	samples random windows from pretrain token cache
	•	mix probs (default: 95% fineweb-edu / 5% roam; wikitext optional)
	•	train/sft.py
	•	samples packed chat sequences; masked loss on assistant tokens
	•	train/run.py
	•	runs: --phase pretrain|sft, or --pipeline pretrain_then_sft
	•	checkpointing: separate “best” per phase (don’t overwrite pretrain best with sft best)

tests
	•	smoke run for each phase (like 200 steps) completes and saves ckpt

⸻

pr-07: mixed precision on mps + grad accumulation (fit the bigger config)

goal: make v1 train fast enough on m4 and not OOM.

deliverables
	•	amp autocast support on mps (fp16), with safe fallback to fp32  ￼
	•	gradient accumulation knobs:
	•	micro_B and accum_steps
	•	optional activation checkpointing toggle (only if needed to fit memory)

tests
	•	one training step on mps (if available) runs
	•	accumulation equivalence sanity: accum_steps=2 roughly matches B*2 gradient direction on a tiny toy batch

⸻

pr-08: inference v1 (kv-cache) + keep your streaming inspector

goal: fast token streaming + better long responses, without changing model weights.

deliverables
	•	infer/kv_cache.py:
	•	cached K/V per layer, append one token at a time
	•	“prefill” vs “decode” path
	•	generation uses <|eot|> stop id only
	•	wire kv-cache signals into your SSE stream (cache length, per-step stats)

tests
	•	kv-cache decoding matches non-cached decoding (same seed, greedy decoding)
	•	stop-on-eot works reliably

pr-09: replace wikitext-centric training with streaming fineweb-edu pretrain + sft