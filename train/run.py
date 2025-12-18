from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from niels_gpt.device import get_device
from niels_gpt.profiles import get_profile, format_profile_list

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
    parser = argparse.ArgumentParser(
        description="niels-gpt training runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Use a named profile
  python -m train.run --phase pretrain --profile smoke-pretrain
  python -m train.run --phase sft --profile dev-sft
  python -m train.run --phase pipeline --profile pipeline-dev

  # Use a custom config file
  python -m train.run --phase pretrain --config configs/my-custom.json

  # List all available profiles
  python -m train.run --list-profiles

  # Resume from checkpoint
  python -m train.run --phase pretrain --profile dev --resume runs/my-run/checkpoints/latest.pt
        """,
    )
    parser.add_argument("--phase", choices=["pretrain", "sft", "pipeline"], help="training phase")
    parser.add_argument(
        "--profile",
        help="named profile (e.g., 'smoke-pretrain', 'dev', 'prod'). Use --list-profiles to see all options.",
    )
    parser.add_argument("--config", help="path to custom config json (overrides --profile)")
    parser.add_argument("--device", default=None, help="cpu or mps (default: auto)")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--no-resume", action="store_true", help="disable auto-resume from checkpoints/latest.pt")
    parser.add_argument("--print_config", action="store_true", help="print resolved settings and exit")
    parser.add_argument("--list-profiles", action="store_true", help="list all available configuration profiles")
    args = parser.parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        print(format_profile_list())
        return

    # Validate required arguments
    if not args.phase:
        parser.error("--phase is required (unless using --list-profiles)")
    if not args.profile and not args.config:
        parser.error("either --profile or --config is required")

    # Resolve config path from profile or explicit path
    if args.config:
        config_path = args.config
    else:
        try:
            profile = get_profile(args.profile)
            config_path = str(profile.path)
            print(f"using profile: {profile.name}")
            print(f"  → {profile.description}")
        except KeyError as e:
            print(f"error: {e}")
            sys.exit(1)

    device = _validate_device(args.device) or get_device()
    print(f"using device: {device}")

    try:
        if args.phase == "pretrain":
            cfg = load_pretrain_job_config(config_path)
            if args.print_config:
                print(json.dumps(cfg.resolved.settings.model_dump(), indent=2))
                return
            run_pretrain(
                cfg,
                device=device,
                resume_path=args.resume,
                no_auto_resume=args.no_resume,
            )
        elif args.phase == "sft":
            cfg = load_sft_job_config(config_path)
            if args.print_config:
                print(json.dumps(cfg.resolved.settings.model_dump(), indent=2))
                return
            run_sft(
                cfg,
                device=device,
                resume_path=args.resume,
                no_auto_resume=args.no_resume,
            )
        else:  # pipeline
            pipeline_cfg = load_pipeline_config(config_path)
            pretrain_cfg_path = pipeline_cfg.pretrain_config_path
            sft_cfg_path = pipeline_cfg.sft_config_path

            if args.print_config:
                pre_cfg = load_pretrain_job_config(pretrain_cfg_path)
                sft_cfg = load_sft_job_config(sft_cfg_path)
                print(
                    json.dumps(
                        {"pretrain": pre_cfg.resolved.settings.model_dump(), "sft": sft_cfg.resolved.settings.model_dump()},
                        indent=2,
                    )
                )
                return

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

