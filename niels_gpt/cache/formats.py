"""Cache format constants and utilities."""

# Token data type
TOKEN_DTYPE = "uint16-le"  # little-endian uint16
TOKEN_BYTES = 2  # 2 bytes per token

# Shard size target for pretrain caches
DEFAULT_SHARD_BYTES = 128 * 1024 * 1024  # 128 MB
