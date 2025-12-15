import torch


def encode(text: str) -> torch.LongTensor:
    """
    utf-8 encode to bytes; each byte is a token id.
    returns: 1D int64 tensor on cpu, shape (n,), values in [0..255].
    """
    byte_values = text.encode("utf-8")
    return torch.tensor(list(byte_values), dtype=torch.long)


def decode(ids: torch.LongTensor) -> str:
    """
    decode ids -> bytes -> utf-8 string for display.
    rule: bytes(ids).decode("utf-8", errors="replace")

    validation:
    - ids must be a torch.Tensor
    - ids must be 1D
    - all values must be ints in [0..255]
    otherwise raise ValueError.
    """
    # Validate that ids is a torch.Tensor
    if not isinstance(ids, torch.Tensor):
        raise ValueError("ids must be a torch.Tensor")

    # Validate that ids is 1D
    if ids.dim() != 1:
        raise ValueError("ids must be 1D")

    # Validate that all values are in [0..255]
    if ids.numel() > 0:
        if ids.min() < 0 or ids.max() > 255:
            raise ValueError("all values must be ints in [0..255]")

    # Convert to bytes and decode
    byte_list = ids.tolist()
    byte_data = bytes(byte_list)
    return byte_data.decode("utf-8", errors="replace")
