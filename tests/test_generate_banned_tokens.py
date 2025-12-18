import torch

from niels_gpt.generate import generate_ids


def test_banned_tokens_masking():
    vocab = 10
    banned = [2, 3]
    eot_id = 9

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Force logits high on banned ids if unmasked
            B, T = x.shape
            logits = torch.zeros(B, T, vocab)
            logits[:, -1, 2] = 100.0
            logits[:, -1, 3] = 100.0
            logits[:, -1, eot_id] = 50.0
            return logits

    model = Dummy()
    prompt = torch.tensor([1, 1], dtype=torch.long)
    out = generate_ids(
        model,
        prompt,
        max_new_tokens=1,
        T=8,
        temperature=0,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        eot_id=eot_id,
        banned_token_ids=banned,
        device="cpu",
        generator=None,
    )
    # Should choose eot_id because banned ids are -inf
    assert out[-1].item() == 1  # prompt preserved
    # Next token appended but truncated on eot, so output length == prompt length

