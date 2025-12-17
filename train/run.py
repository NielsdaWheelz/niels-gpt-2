from __future__ import annotations

import argparse
from pathlib import Path
import sys

from niels_gpt.device import get_device

from train.config import (
    load_pipeline_config,
    load_pretrain_job_config,
    load_sft_job_config,
)
from train.pretrain import run_pretrain
from train.sft import run_sft


def _validate_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device not in {"cpu", "mps"}:
        raise ValueError("--device must be one of {cpu,mps}")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="pr-06 training runner")
    parser.add_argument("--phase", required=True, choices=["pretrain", "sft", "pipeline"])
    parser.add_argument("--config", required=True, help="path to phase config json")
    parser.add_argument("--device", default=None, help="cpu or mps (default: auto)")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--no-resume", action="store_true", help="disable auto-resume from checkpoints/latest.pt")
    args = parser.parse_args()

    device = _validate_device(args.device) or get_device()
    print(f"using device: {device}")

    try:
        if args.phase == "pretrain":
            cfg = load_pretrain_job_config(args.config)
            run_pretrain(
                cfg,
                device=device,
                resume_path=args.resume,
                no_auto_resume=args.no_resume,
            )
        elif args.phase == "sft":
            cfg = load_sft_job_config(args.config)
            run_sft(
                cfg,
                device=device,
                resume_path=args.resume,
                no_auto_resume=args.no_resume,
            )
        else:  # pipeline
            pipeline_cfg = load_pipeline_config(args.config)
            pretrain_cfg_path = pipeline_cfg.pretrain_config_path
            sft_cfg_path = pipeline_cfg.sft_config_path

            print("=== pipeline: pretrain ===")
            pretrain_result = run_pretrain(
                load_pretrain_job_config(pretrain_cfg_path),
                device=device,
                resume_path=args.resume,
                no_auto_resume=args.no_resume,
            )
            init_candidate = pretrain_result.get("best_path")
            if init_candidate is None or not Path(init_candidate).exists():
                init_candidate = pretrain_result.get("latest_path")
            if init_candidate is None or not Path(init_candidate).exists():
                raise RuntimeError("pretrain phase did not produce a checkpoint")

            print("\n=== pipeline: sft (init from pretrain best) ===")
            run_sft(
                load_sft_job_config(sft_cfg_path),
                device=device,
                resume_path=None,
                no_auto_resume=True,
                init_model_path=str(init_candidate),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

