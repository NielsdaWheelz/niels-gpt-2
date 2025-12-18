"""Interactive CLI chat with trained checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

from niels_gpt.chat_format import extract_assistant_reply, format_chat
from niels_gpt.checkpoint import load_checkpoint
from niels_gpt.config import ModelConfig
from niels_gpt.device import get_device
from niels_gpt.generate import generate_ids_cached, generate_text
from niels_gpt.model.gpt import GPT
from niels_gpt.settings import default_settings
from niels_gpt.tokenizer import get_default_tokenizer


DEFAULT_SYSTEM_PROMPT = "you are a surly chatbot for niels' site. answer in third person about niels. be terse, rude but not offensive, grudging, laconic, and brooding. answer like you resent being asked. do not guess. say 'i don't know' when unsure. refuse private info with 'no'."


def main():
    settings = default_settings()
    gen_defaults = settings.generation
    parser = argparse.ArgumentParser(description="Interactive chat CLI")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=gen_defaults.max_new_tokens,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=gen_defaults.temperature,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=gen_defaults.top_k if gen_defaults.top_k is not None else 0,
        help="Top-k filtering (0 for none)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=gen_defaults.top_p if gen_defaults.top_p is not None else 0.0,
        help="Top-p filtering (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--system", type=str, default=None, help="System message content")
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="Read system message from file (default: configs/system_surly.txt if present)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache and use full forward pass per token (slower, for debugging)",
    )

    args = parser.parse_args()

    # Determine system prompt (precedence: --system-file > --system > default file > builtin)
    default_system_path = Path(__file__).parent.parent / "configs" / "system_surly.txt"
    system_path = args.system_file or (str(default_system_path) if default_system_path.exists() else None)
    if system_path:
        with open(system_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    elif args.system:
        system_prompt = args.system
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Handle top-k (0 means None)
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None

    gen_overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    generation_cfg = gen_defaults.model_copy(update=gen_overrides)
    print("generation settings (settings defaults + CLI overrides):")
    print(json.dumps(generation_cfg.model_dump(), indent=2))

    # Load checkpoint and model
    device = get_device()
    ckpt = load_checkpoint(args.ckpt, device=device)
    model_cfg_dict = {k: v for k, v in ckpt["model_cfg"].items() if k != "_raw"}
    cfg = ModelConfig(**model_cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Create generator for deterministic sampling (always CPU for cross-device compatibility)
    # MPS has inconsistent generator support, so we always use CPU generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    tok = get_default_tokenizer()
    special_ids = tok.special_token_ids()
    stop_id = gen_defaults.stop_token_id or special_ids["eot"]
    banned_ids = gen_defaults.banned_token_ids
    if not banned_ids and settings.sft_format.ban_role_tokens_during_generation:
        banned_ids = [special_ids["sys"], special_ids["usr"], special_ids["asst"]]

    # Initialize messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Interaction loop
    try:
        while True:
            # Read user input
            try:
                user_input = input("> ")
            except EOFError:
                # Clean exit on ctrl-d
                break

            # Skip empty lines
            if not user_input:
                continue

            # Exit command
            if user_input == "/exit":
                break

            # Append user message
            messages.append({"role": "user", "content": user_input})

            # Format chat prompt
            prompt = format_chat(messages)

            # Generate response
            if args.no_kv_cache:
                # Use uncached generation (full forward pass per token)
                generated_text = generate_text(
                    model,
                    prompt,
                    cfg=cfg,
                    generation=generation_cfg,
                    stop_token_id=stop_id,
                    banned_token_ids=banned_ids,
                    device=device,
                    generator=generator,
                )
                reply = extract_assistant_reply(generated_text)
            else:
                # Use KV-cache generation (default, faster)
                prompt_ids = tok.encode(prompt)
                result = generate_ids_cached(
                    model,
                    prompt_ids,
                    max_new_tokens=generation_cfg.max_new_tokens,
                    eot_token_id=stop_id,
                    temperature=generation_cfg.temperature,
                    top_k=generation_cfg.top_k,
                    top_p=generation_cfg.top_p,
                    repetition_penalty=generation_cfg.repetition_penalty,
                    banned_token_ids=banned_ids,
                    generator=generator,
                )
                generated_text = tok.decode(result["ids"])
                reply = extract_assistant_reply(generated_text)

            # Print reply
            print(reply)

            # Append assistant message
            messages.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        # Clean exit on ctrl-c
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
