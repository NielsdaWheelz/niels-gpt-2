import tempfile
from pathlib import Path

import numpy as np
import torch

from train.pretrain import run_pretrain
from train.sft import run_sft
from train.checkpointing import load_checkpoint
import niels_gpt.paths as ng_paths
from niels_gpt.cache.sft_dataset import SFTExampleDataset
from niels_gpt.cache.meta import sha256_file
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer


def _write_stream_cache(base: Path, source: str, split: str, tokens: list[int], special_ids: dict[str, int] | None = None) -> None:
    source_dir = base / source
    split_dir = source_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    bin_path = split_dir / "shard_00000.bin"
    meta_path = source_dir / "meta.json"
    arr = np.asarray(tokens, dtype="<u2")
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())
    tokenizer_sha = sha256_file(str(DEFAULT_TOKENIZER_PATH))
    if special_ids is None:
        special_ids = load_tokenizer(str(DEFAULT_TOKENIZER_PATH)).special_token_ids()
    meta = {
        "token_dtype": "uint16-le",
        "source": source,
        "split": split,
        "tokenizer_sha256": tokenizer_sha,
        "special_token_ids": special_ids,
    }
    meta_path.write_text(json_dumps(meta))


def _write_sft_cache(base: Path, source: str, split: str, sequences: list[list[int]], special: dict[str, int]) -> None:
    source_dir = base / source
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / split).mkdir(exist_ok=True)

    tokens_path = source_dir / f"{split}_input_ids.bin"
    labels_path = source_dir / f"{split}_labels.bin"
    idx_path = source_dir / f"{split}_idx.npy"
    meta_path = source_dir / "meta.json"

    offsets: list[int] = []
    pos = 0
    with open(tokens_path, "wb") as f, open(labels_path, "wb") as lf:
        for seq in sequences:
            offsets.append(pos)
            arr = np.asarray(seq, dtype="<u2")
            lbl = np.asarray(seq, dtype="<i4")  # unmasked targets for tests
            f.write(arr.tobytes())
            lf.write(lbl.tobytes())
            pos += len(seq)

    np.save(idx_path, np.asarray(offsets, dtype=np.int64))
    tokenizer_sha = sha256_file(str(DEFAULT_TOKENIZER_PATH))
    meta = {
        "token_dtype": "uint16-le",
        "label_dtype": "int32-le",
        "special_token_ids": special,
        "source": source,
        "split": split,
        "tokenizer_sha256": tokenizer_sha,
    }
    meta_path.write_text(json_dumps(meta))


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def _patch_checkpoints(temp_dir: Path):
    orig = ng_paths.CHECKPOINT_DIR
    ng_paths.CHECKPOINT_DIR = temp_dir
    return orig


def test_pretrain_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        streams_dir = tmp_path / "streams"
        ckpt_dir = tmp_path / "ckpts"
        orig_ckpt = _patch_checkpoints(ckpt_dir)
        try:
            sources = {
                "wikitext": list(range(100, 150)),
                "roam": list(range(200, 260)),
            }
            for src, toks in sources.items():
                _write_stream_cache(streams_dir, src, "train", toks)
                _write_stream_cache(streams_dir, src, "val", toks)

            cfg = {
                "model": {
                    "V": 512,
                    "T": 8,
                    "C": 32,
                    "L": 1,
                    "H": 4,
                    "d_ff": 64,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "training": {
                    "pretrain": {
                        "micro_B": 2,
                        "total_steps": 6,
                        "eval_every": 2,
                        "eval_steps": 2,
                        "ckpt_every": 3,
                        "base_lr": 0.001,
                        "warmup_steps": 1,
                        "min_lr": 1e-5,
                        "grad_clip": 1.0,
                        "accum_steps": 1,
                        "log_every": 1,
                        "seed": 42,
                    }
                },
                "data": {
                    "mix_pretrain": {"wikitext": 0.7, "roam": 0.3},
                    "val_pretrain_source": "wikitext",
                    "caches": {"pretrain_token_cache": str(streams_dir)},
                },
            }

            result = run_pretrain(cfg, device="cpu", no_auto_resume=True)

            assert result["latest_path"].exists()
            assert (ckpt_dir / "pretrain" / "step_0000003.pt").exists()
            assert result["best_val_loss"] is not None
            assert np.isfinite(result["best_val_loss"])
        finally:
            ng_paths.CHECKPOINT_DIR = orig_ckpt


def test_resume_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        streams_dir = tmp_path / "streams"
        ckpt_dir = tmp_path / "ckpts"
        orig_ckpt = _patch_checkpoints(ckpt_dir)
        try:
            tokens = list(range(100, 200))
            for src in ["wikitext", "roam"]:
                _write_stream_cache(streams_dir, src, "train", tokens)
                _write_stream_cache(streams_dir, src, "val", tokens)

            base_cfg = {
                "model": {
                    "V": 256,
                    "T": 8,
                    "C": 32,
                    "L": 1,
                    "H": 4,
                    "d_ff": 64,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "data": {
                    "mix_pretrain": {"wikitext": 0.7, "roam": 0.3},
                    "val_pretrain_source": "wikitext",
                    "caches": {"pretrain_token_cache": str(streams_dir)},
                },
            }

            cfg_first = {
                **base_cfg,
                "training": {
                    "pretrain": {
                        "micro_B": 2,
                        "total_steps": 5,
                        "eval_every": 5,
                        "eval_steps": 1,
                        "ckpt_every": 5,
                        "base_lr": 0.001,
                        "warmup_steps": 0,
                        "min_lr": 1e-5,
                        "grad_clip": 1.0,
                        "accum_steps": 1,
                        "log_every": 1,
                        "seed": 1,
                    }
                },
            }
            first = run_pretrain(cfg_first, device="cpu", no_auto_resume=True)
            assert first["latest_path"].exists()

            cfg_resume = {
                **base_cfg,
                "training": {
                    "pretrain": {
                        **cfg_first["training"]["pretrain"],
                        "total_steps": 10,
                        "eval_every": 5,
                        "ckpt_every": 5,
                    }
                },
            }
            run_pretrain(
                cfg_resume,
                device="cpu",
                resume_path=str(first["latest_path"]),
                no_auto_resume=True,
            )

            latest = load_checkpoint(str(ckpt_dir / "pretrain" / "latest.pt"), device="cpu")
            assert latest["step"] == 10
            assert (ckpt_dir / "pretrain" / "step_0000010.pt").exists()
        finally:
            ng_paths.CHECKPOINT_DIR = orig_ckpt


def test_sft_masking_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sft_dir = tmp_path / "sft"
        streams_dir = tmp_path / "streams"
        ckpt_dir = tmp_path / "ckpts"
        orig_ckpt = _patch_checkpoints(ckpt_dir)
        try:
            special = {"sys": 3, "usr": 4, "asst": 5, "eot": 6}
            seq = [
                special["sys"],
                10,
                special["eot"],
                special["usr"],
                11,
                special["eot"],
                special["asst"],
                20,
                21,
                special["eot"],
            ]
            _write_sft_cache(sft_dir, "dolly15k", "train", [seq], special)
            _write_sft_cache(sft_dir, "dolly15k", "val", [seq], special)

            ds = SFTExampleDataset(
                str(sft_dir / "dolly15k" / "train_input_ids.bin"),
                str(sft_dir / "dolly15k" / "train_idx.npy"),
                str(sft_dir / "dolly15k" / "train_labels.bin"),
                T=len(seq) - 1,
                device="cpu",
                eot_id=special["eot"],
                asst_id=special["asst"],
            )
            gen = torch.Generator().manual_seed(0)
            x, _, y_masked = ds.get_batch(B=1, generator=gen)

            V = 32
            logits = torch.zeros((1, x.shape[1], V))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V), y_masked.view(-1), ignore_index=-100
            )

            mask = (y_masked.view(-1) != -100)
            targets = y_masked.view(-1)[mask]
            manual = torch.nn.functional.cross_entropy(
                torch.zeros((targets.numel(), V)), targets
            )
            assert torch.allclose(loss, manual, atol=1e-6)

            cfg = {
                "model": {
                    "V": V,
                    "T": len(seq) - 1,
                    "C": 16,
                    "L": 1,
                    "H": 2,
                    "d_ff": 32,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "training": {
                    "sft": {
                        "micro_B": 1,
                        "total_steps": 1,
                        "eval_every": 1,
                        "eval_steps": 1,
                        "ckpt_every": 1,
                        "base_lr": 0.001,
                        "warmup_steps": 0,
                        "min_lr": 0.001,
                        "grad_clip": 1.0,
                        "accum_steps": 1,
                        "log_every": 1,
                        "seed": 0,
                    }
                },
                "data": {
                    "mix_sft": {"dolly15k": 1.0},
                    "val_sft_source": "sft",
                    "caches": {
                        "sft_token_cache": str(sft_dir),
                        # wikitext val uses the pretrain cache, not streams_token_cache
                        "pretrain_token_cache": str(streams_dir),
                    },
                },
            }

            # Provide wikitext cache (now used for val_pretrain_loss in pr-02)
            _write_stream_cache(streams_dir, "wikitext", "val", [i % V for i in range(20)], special_ids=special)
            result = run_sft(cfg, device="cpu", no_auto_resume=True)
            assert (ckpt_dir / "latest.pt").exists()
            assert result["last_val_pretrain_loss"] is not None
            assert np.isfinite(result["last_val_pretrain_loss"])
            assert result["last_val_sft_loss"] is not None
            assert np.isfinite(result["last_val_sft_loss"])
            assert result["best_val_loss"] is not None
            assert np.isfinite(result["best_val_loss"])
        finally:
            ng_paths.CHECKPOINT_DIR = orig_ckpt

