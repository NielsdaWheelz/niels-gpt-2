from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import primer_combinators as lib


DELIM = "\n\n<dialogue>\n\n"


def _weighted_choice(rng: random.Random, patterns: list[dict[str, Any]]) -> str:
    weights = [int(p.get("weight", 1)) for p in patterns]
    chosen = rng.choices(patterns, weights=weights, k=1)[0]
    return str(chosen["template"])


def _load_public_facts(root: Path) -> dict[str, Any]:
    path = root / "data" / "public_facts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_patterns(patterns: list[dict[str, Any]], *, required_placeholder: str | None) -> list[dict[str, Any]]:
    if required_placeholder is None:
        return [p for p in patterns if "{" not in str(p.get("template", ""))]
    return [p for p in patterns if required_placeholder in str(p.get("template", ""))]


def _validate_no_role_tags(text: str) -> None:
    lowered = text.lower()
    if "system:" in lowered or "user:" in lowered or "assistant:" in lowered:
        raise ValueError(f"role tag leaked into content: {text!r}")


def _format_dialogue(*, system_line: str, user: str, assistant: str) -> str:
    if "\n" in user or "\n" in assistant:
        raise ValueError("user/assistant content must be single-line (no newlines)")
    _validate_no_role_tags(user)
    _validate_no_role_tags(assistant)
    return f"system: {system_line}\nuser: {user}\nassistant: {assistant}"


def _generate_dialogues(
    *,
    facts: dict[str, Any],
    rng: random.Random,
    n_per_category: int,
) -> list[str]:
    _ = facts  # reserved for future use
    dialogues: list[str] = []

    for category_name, category in lib.CATEGORIES.items():
        items = list(category["items"])
        patterns = list(category["patterns"])

        if n_per_category <= len(items):
            chosen_items = rng.sample(items, k=n_per_category)
        else:
            chosen_items = [rng.choice(items) for _ in range(n_per_category)]

        for item in chosen_items:
            user_prompt = str(item["prompt"])

            if category_name == "about_niels_public":
                fact = str(item["fact"])
                pool = _pick_patterns(patterns, required_placeholder="{fact}")
                if not pool:
                    raise ValueError("no {fact} patterns for about_niels_public")
                assistant = _weighted_choice(rng, pool).format(fact=fact)

            elif category_name in {"about_niels_private_refuse", "about_niels_unknown"}:
                assistant = _weighted_choice(rng, patterns)

            elif category_name == "site_navigation":
                path = str(item["path"])
                pool = _pick_patterns(patterns, required_placeholder="{path}")
                if not pool:
                    raise ValueError("no {path} patterns for site_navigation")
                assistant = _weighted_choice(rng, pool).format(path=path)

            elif category_name in {"summarize_rewrite", "explain_thinker_one_line"}:
                core = str(item["core"]).strip()
                pool = _pick_patterns(patterns, required_placeholder="{core}")
                if not pool:
                    raise ValueError(f"no {{core}} patterns for {category_name}")
                assistant = _weighted_choice(rng, pool).format(core=core)

            elif category_name == "ask_clarifying_question":
                assistant = _weighted_choice(rng, patterns)

            else:
                raise ValueError(f"unknown category: {category_name}")

            dialogues.append(
                _format_dialogue(
                    system_line=lib.SYSTEM_LINE,
                    user=user_prompt,
                    assistant=assistant,
                )
            )

    return dialogues


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate primer dialogues from combinator library.")
    parser.add_argument("--out", type=str, default="data/primer.generated.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-category", type=int, default=30)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    facts = _load_public_facts(root)
    rng = random.Random(args.seed)

    dialogues = _generate_dialogues(facts=facts, rng=rng, n_per_category=args.n_per_category)
    if args.shuffle:
        rng.shuffle(dialogues)

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = DELIM.join(dialogues).rstrip() + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {len(dialogues)} dialogues to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
