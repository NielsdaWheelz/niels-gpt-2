"""Test SFT masking correctness."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from niels_gpt.cache.sft_dataset import SFTExampleDataset


def test_sft_masking():
    """
    Test that SFT masking correctly identifies assistant spans.

    Hand-construct a sequence with:
    <|sys|> ... <|eot|> <|usr|> ... <|eot|> <|asst|> A B <|eot|>

    Verify that only targets for A, B, and <|eot|> are not -100.
    """
    # Define special token IDs
    sys_id = 100
    usr_id = 101
    asst_id = 102
    eot_id = 103

    # Content token IDs
    sys_content = [10, 11]  # system message content
    usr_content = [20, 21]  # user message content
    asst_content = [30, 31]  # assistant message content (A, B)

    # Construct sequence:
    # <sys> 10 11 <eot> <usr> 20 21 <eot> <asst> 30 31 <eot>
    sequence = (
        [sys_id] + sys_content + [eot_id]
        + [usr_id] + usr_content + [eot_id]
        + [asst_id] + asst_content + [eot_id]
    )

    # Create temporary cache files
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / "tokens.bin"
        idx_path = Path(tmpdir) / "idx.npy"

        # Write tokens (uint16 little-endian)
        with open(tokens_path, "wb") as f:
            for token in sequence:
                f.write(token.to_bytes(2, byteorder="little"))

        # Write offsets (single example starting at 0)
        np.save(idx_path, np.array([0], dtype=np.int64))

        # Create dataset
        T = len(sequence) - 1  # Use full sequence length
        dataset = SFTExampleDataset(
            str(tokens_path),
            str(idx_path),
            T=T,
            device="cpu",
            eot_id=eot_id,
            asst_id=asst_id,
        )

        # Get a batch
        gen = torch.Generator().manual_seed(42)
        x, y, y_masked = dataset.get_batch(B=1, generator=gen)

        # Expected behavior:
        # x should be sequence[:-1]
        # y should be sequence[1:]
        # y_masked should have -100 for all positions except assistant span

        # Assistant span starts after <asst> token
        # In x, <asst> is at position where sequence[pos] == asst_id
        # Targets after that position should not be -100

        # Let's find where asst_id appears in the sequence
        seq_array = np.array(sequence)
        asst_positions = np.where(seq_array == asst_id)[0]

        # There should be one <asst> token
        assert len(asst_positions) == 1
        asst_pos = asst_positions[0]

        # In y_masked, positions corresponding to assistant content should not be -100
        # Position i in y corresponds to predicting sequence[i+1] given sequence[:i+1]
        # If x[i] == asst_id, then we start predicting assistant content
        # So y[i] (which is sequence[i+1]) should not be -100

        # Actually, let's think about this more carefully:
        # - x = sequence[:-1]
        # - y = sequence[1:]
        # - For masking, we want to predict tokens that come after <asst> up to and including <eot>

        # The assistant span in the sequence is: <asst> 30 31 <eot>
        # In x, this appears at positions [asst_pos, asst_pos+1, asst_pos+2, asst_pos+3]
        # In y, we're predicting [30, 31, <eot>, next_token]

        # We want to compute loss on [30, 31, <eot>]
        # These are at positions [asst_pos, asst_pos+1, asst_pos+2] in y

        y_masked_array = y_masked[0].cpu().numpy()

        # Check that assistant content positions are NOT -100
        for offset in range(len(asst_content) + 1):  # +1 for <eot>
            pos = asst_pos + offset
            if pos < len(y_masked_array):
                assert y_masked_array[pos] != -100, (
                    f"Position {pos} should not be -100 (assistant span), "
                    f"but got {y_masked_array[pos]}"
                )

        # Check that non-assistant positions ARE -100
        for pos in range(asst_pos):
            assert y_masked_array[pos] == -100, (
                f"Position {pos} should be -100 (not assistant), "
                f"but got {y_masked_array[pos]}"
            )

        # Check that positions after assistant span are -100
        after_asst = asst_pos + len(asst_content) + 1
        for pos in range(after_asst, len(y_masked_array)):
            assert y_masked_array[pos] == -100, (
                f"Position {pos} should be -100 (after assistant span), "
                f"but got {y_masked_array[pos]}"
            )

    print("✓ SFT masking test passed")


if __name__ == "__main__":
    test_sft_masking()
