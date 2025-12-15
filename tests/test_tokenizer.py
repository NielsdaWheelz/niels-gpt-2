import torch

from niels_gpt.tokenizer import decode, encode


def test_encode_basic_ascii():
    result = encode("A")
    assert torch.equal(result, torch.tensor([65]))


def test_encode_empty():
    result = encode("")
    assert result.shape == (0,)
    assert result.dtype == torch.int64


def test_decode_basic_ascii():
    result = decode(torch.tensor([65, 66]))
    assert result == "AB"


def test_decode_empty():
    empty_tensor = torch.tensor([], dtype=torch.long)
    result = decode(empty_tensor)
    assert result == ""


def test_decode_rejects_non_1d():
    try:
        decode(torch.zeros((2, 2), dtype=torch.long))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_decode_rejects_out_of_range():
    # Test value > 255
    try:
        decode(torch.tensor([256], dtype=torch.long))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test value < 0
    try:
        decode(torch.tensor([-1], dtype=torch.long))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
