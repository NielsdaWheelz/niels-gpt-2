# charter: tiny llm-from-scratch (macbook air m4)

## goal
build a small language model from scratch: i write the transformer code, i train the weights, and i can explain every part.

deliver a minimal “chat” interface (text in, text out) locally, then host it on my personal website.

## non-goals
- frontier / sota capability
- safety tuning, rlhf, tool use, retrieval, long context
- production-grade scaling, moderation, or security hardening
- data cleaning, filtering, regex pipelines, or dataset engineering

## constraints
- hardware: single macbook air (apple silicon m4)
- accelerator: use pytorch “mps” (mac gpu) when available; fall back to cpu if needed
- training time: ≤ 6 hours for the main training run
- tokenizer: byte-level (each byte is a token; vocab size = 256)
- publishing: code + weights only (no redistribution of training text)

personal corpus:
- roam export as a folder of `.md` files (~13mb)
- ingestion rule: read all `.md` files and join with a fixed separator

public corpus (packaged):
- primary: wikitext-103 (download via a dataset loader)
- optional later: pg-19

## model scope (keep it small first)
architecture:
- decoder-only transformer (gpt-style)
- causal self-attention (can only look at earlier tokens)
- rotary position embeddings (RoPE) for better length extrapolation

training objective:
- next-token prediction (predict the next byte given previous bytes)

“instruction-following feel” (lightweight, not true alignment):
- format prompts as chat transcripts:
  - system: ...
  - user: ...
  - assistant: ...
- include a small handwritten “chat primer” text (tens of examples) mixed into training
  to bias the model toward producing “assistant:” continuations

size constraint:
- “whatever fits on mps without crashing”
- start small; only scale up after the end-to-end pipeline is proven

## definition of done
training correctness:
- can overfit a tiny sample (sanity check) to near-zero loss
- full run: training loss decreases and validation loss decreases at least initially

inference:
- can generate text from a prompt
- can run an interactive cli chat loop (maintains conversation transcript)

web:
- minimal endpoint `/chat` that accepts a transcript and returns generated text
- minimal web page that posts messages and renders the transcript

explainability:
- i can explain attention with tensor shapes:
  - input -> q/k/v -> attention scores -> causal mask -> softmax -> weighted sum -> output
- i can explain why we need the mask, why we use softmax, and what the loss means.

## milestones
1) environment check: mps works and a tensor op runs on it
2) tokenizer + data loader + unit tests
3) transformer forward pass + causal mask unit test
4) training loop + overfit test
5) full training run (≤ 6h)
6) sampling + cli chat
7) web chat endpoint + minimal ui

## risks (expected failure modes)
- small model + limited compute ⇒ shallow coherence, repetition, weak instruction following
- overfitting to roam style if public corpus is too small
- mps memory limits forcing smaller batch/model than hoped