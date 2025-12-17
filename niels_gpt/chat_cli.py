"""Interactive CLI chat with trained checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

from niels_gpt.chat_format import extract_assistant_reply, format_chat
from niels_gpt.checkpoint import load_checkpoint
from niels_gpt.config import ModelConfig
from niels_gpt.device import get_device
from niels_gpt.generate import generate_text
from niels_gpt.model.gpt import GPT


DEFAULT_SYSTEM_PROMPT = "you are a surly chatbot for niels' site. answer in third person about niels. be terse, rude but not offensive, grudging, laconic, and brooding. answer like you resent being asked. do not guess. say 'i don't know' when unsure. refuse private info with 'no'."


def main():
    parser = argparse.ArgumentParser(description="Interactive chat CLI")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (0 for none)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--system", type=str, default=None, help="System message content")
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="Read system message from file (default: configs/system_surly.txt if present)",
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
            generated_text = generate_text(
                model,
                prompt,
                cfg=cfg,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                device=device,
                generator=generator,
            )

            # Extract assistant reply
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
