"""Test SFT masking correctness with cached labels."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from niels_gpt.cache.sft_dataset import SFTExampleDataset


def test_sft_masking():
    sys_id = 100
    usr_id = 101
    asst_id = 102
    eot_id = 103

    sys_content = [10, 11]
    usr_content = [20, 21]
    asst_content = [30, 31]

    sequence = (
        [sys_id] + sys_content + [eot_id]
        + [usr_id] + usr_content + [eot_id]
        + [asst_id] + asst_content + [eot_id]
    )

    labels = []
    in_assistant = False
    for tok in sequence:
        if tok == asst_id:
            labels.append(-100)
            in_assistant = True
            continue
        if in_assistant:
            labels.append(tok)
            if tok == eot_id:
                in_assistant = False
        else:
            labels.append(-100)

    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / "input_ids.bin"
        labels_path = Path(tmpdir) / "labels.bin"
        idx_path = Path(tmpdir) / "idx.npy"

        with open(tokens_path, "wb") as f:
            for token in sequence:
                f.write(token.to_bytes(2, byteorder="little"))
        with open(labels_path, "wb") as f:
            for label in labels:
                f.write(int(label).to_bytes(4, byteorder="little", signed=True))
        np.save(idx_path, np.array([0], dtype=np.int64))

        dataset = SFTExampleDataset(
            str(tokens_path),
            str(idx_path),
            str(labels_path),
            T=len(sequence) - 1,
            device="cpu",
            eot_id=eot_id,
            asst_id=asst_id,
        )

        gen = torch.Generator().manual_seed(42)
        _, _, y_masked = dataset.get_batch(B=1, generator=gen)

        expected_labels = np.array(labels[1:], dtype=np.int64)
        np.testing.assert_array_equal(y_masked[0].cpu().numpy(), expected_labels)

