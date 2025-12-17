import tempfile
from pathlib import Path

import numpy as np
import torch

from train.pretrain import run_pretrain
from train.sft import run_sft
from train.checkpointing import load_checkpoint
import niels_gpt.paths as ng_paths
from niels_gpt.cache.sft_dataset import SFTExampleDataset


def _write_stream_cache(base: Path, source: str, split: str, tokens: list[int]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    bin_path = base / f"{source}_{split}.bin"
    meta_path = base / f"{source}_{split}.meta.json"
    arr = np.asarray(tokens, dtype="<u2")
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())
    meta = {"token_dtype": "uint16-le", "source": source, "split": split}
    meta_path.write_text(json_dumps(meta))


def _write_sft_cache(base: Path, source: str, split: str, sequences: list[list[int]], special: dict[str, int]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    tokens_path = base / f"{source}_{split}.bin"
    idx_path = base / f"{source}_{split}.idx.npy"
    meta_path = base / f"{source}_{split}.meta.json"

    offsets: list[int] = []
    pos = 0
    with open(tokens_path, "wb") as f:
        for seq in sequences:
            offsets.append(pos)
            arr = np.asarray(seq, dtype="<u2")
            f.write(arr.tobytes())
            pos += len(seq)

    np.save(idx_path, np.asarray(offsets, dtype=np.int64))
    meta = {
        "token_dtype": "uint16-le",
        "special_token_ids": special,
        "source": source,
        "split": split,
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
                "wiki": list(range(100, 150)),
                "roam": list(range(200, 260)),
                "primer": list(range(300, 360)),
            }
            for src, toks in sources.items():
                _write_stream_cache(streams_dir, src, "train", toks)
                _write_stream_cache(streams_dir, src, "val", toks)

            cfg = {
                "model_cfg": {
                    "V": 512,
                    "T": 8,
                    "C": 32,
                    "L": 1,
                    "H": 4,
                    "d_ff": 64,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "train_cfg": {
                    "B": 2,
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
                },
                "sources": {"wiki": 0.6, "roam": 0.3, "primer": 0.1},
                "val_source": "wiki",
                "cache_dir": str(streams_dir),
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
            for src in ["wiki", "roam", "primer"]:
                _write_stream_cache(streams_dir, src, "train", tokens)
                _write_stream_cache(streams_dir, src, "val", tokens)

            base_cfg = {
                "model_cfg": {
                    "V": 256,
                    "T": 8,
                    "C": 32,
                    "L": 1,
                    "H": 4,
                    "d_ff": 64,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "sources": {"wiki": 0.7, "roam": 0.2, "primer": 0.1},
                "val_source": "wiki",
                "cache_dir": str(streams_dir),
            }

            cfg_first = {
                **base_cfg,
                "train_cfg": {
                    "B": 2,
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
                },
            }
            first = run_pretrain(cfg_first, device="cpu", no_auto_resume=True)
            assert first["latest_path"].exists()

            cfg_resume = {
                **base_cfg,
                "train_cfg": {
                    **cfg_first["train_cfg"],
                    "total_steps": 10,
                    "eval_every": 5,
                    "ckpt_every": 5,
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
            special = {"sys": 1, "usr": 2, "asst": 3, "eot": 4}
            seq = [1, 10, 4, 2, 11, 4, 3, 20, 21, 4]
            _write_sft_cache(sft_dir, "dolly", "train", [seq], special)
            _write_sft_cache(sft_dir, "dolly", "val", [seq], special)

            ds = SFTExampleDataset(
                str(sft_dir / "dolly_train.bin"),
                str(sft_dir / "dolly_train.idx.npy"),
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
                "model_cfg": {
                    "V": V,
                    "T": len(seq) - 1,
                    "C": 16,
                    "L": 1,
                    "H": 2,
                    "d_ff": 32,
                    "dropout": 0.1,
                    "rope_theta": 10000.0,
                },
                "train_cfg": {
                    "B": 1,
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
                },
                "sft_sources": {"dolly": 1.0},
                "val_source": "sft",
                "cache_dir": str(sft_dir),
                "streams_cache_dir": str(streams_dir),
            }

            # Provide dummy wiki cache for completeness even if unused
            _write_stream_cache(streams_dir, "wiki", "val", list(range(20)))
            run_sft(cfg, device="cpu", no_auto_resume=True)
            assert (ckpt_dir / "latest.pt").exists()
        finally:
            ng_paths.CHECKPOINT_DIR = orig_ckpt

