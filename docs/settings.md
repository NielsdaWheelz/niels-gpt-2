# settings and overrides

- defaults live in `niels_gpt/settings.py` (single source of truth for tokenizer, data, model, training, generation, benchmark, reproducibility).
- resolved configs come from `resolve_settings(phase, overrides)`; overrides must be Settings-shaped JSON/dict (no legacy translation). `data.mix_pretrain|mix_sft` are replace-only; other dicts deep-merge.
- each `resolve_settings(..., write_resolved=True)` run writes `runs/<run_id>/resolved_settings.json` and checkpoint sidecars carry `_settings_meta` pointers.
- training entrypoints (`python -m train.run --phase pretrain|sft|pipeline`) treat `--config` as overrides; print resolved config via `--print_config`.
- generation defaults (stop on the EOT sentinel, role-token bans) and benchmark grids are pulled from settings; update settings to change behavior.
- special tokens (`<|ngpt_sys_84a5023f67d74cf29cc4001becde983c|>`, `<|ngpt_usr_84a5023f67d74cf29cc4001becde983c|>`, `<|ngpt_asst_84a5023f67d74cf29cc4001becde983c|>`, `<|ngpt_eot_84a5023f67d74cf29cc4001becde983c|>`) are trained as user_defined_symbols and must encode/decode as single pieces; loaders hard-fail otherwise.
- tokenizer training should include representative corpora (roam, primer, wikitext, fineweb-edu sample) and uses byte_fallback; collision guard rejects any raw text containing special token strings before tokenization.
- cache metadata carries tokenizer sha256; loaders hard-fail on hash mismatch to prevent mixing caches across tokenizers.
- audit guardrail: `python tools/audit_config_coverage.py` allows hyperparam literals only in settings; flags denylisted names and special-token literals elsewhere. Canary tests live in `tests/test_audit_config_coverage.py`.

