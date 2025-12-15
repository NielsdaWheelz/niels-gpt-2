### niels-gpt

tiny llm-from-scratch project + a small chat primer.

### primer format

- **file**: `data/primer.txt`
- **turn format**: `system: …`, `user: …`, `assistant: …` (one line each)
- **dialogue delimiter**: literal `\n\n<dialogue>\n\n` between dialogues
- **system line**: identical in every dialogue (repetition is intentional)
- **no tag leakage**: assistant content should not contain `system:` / `user:` / `assistant:`

### generate “vast” dialogues (optional)

edit prompts + response patterns in `tools/primer_combinators.py`, then run:

```bash
python3 tools/generate_primer.py --seed 0 --n-per-category 30 --shuffle
```

this writes `data/primer.generated.txt`.
