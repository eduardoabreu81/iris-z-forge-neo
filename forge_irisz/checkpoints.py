# Iris-Z support for Forge Neo
# Whether a checkpoint file is an Iris-Z, from its header alone (no weights read). Read before Forge loads the
# checkpoint, by the generation guards: p.sd_model is still the previous model in before_process.

import json
import os
import struct
from functools import lru_cache

from .native import detect

MAX_HEADER_BYTES = 64 * 1024 * 1024
UNET_PREFIXES = ("", "model.diffusion_model.")


def is_iris_z(names) -> bool:
    return any(detect.is_iris_z(names, prefix) for prefix in UNET_PREFIXES)


def _safetensors_names(stream, size):
    length = struct.unpack("<Q", stream.read(8))[0]
    if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
        raise ValueError("invalid safetensors header")
    header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return set(header)


@lru_cache(maxsize=32)
def _file_is_iris_z(path: str, size: int, mtime: int) -> bool:
    try:
        with open(path, "rb") as stream:
            return is_iris_z(_safetensors_names(stream, size))
    except (OSError, UnicodeError, ValueError, struct.error):
        return False


def file_is_iris_z(path) -> bool:
    """For a .safetensors checkpoint, cached by size and modification time."""
    if not str(path).lower().endswith(".safetensors"):
        return False
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return _file_is_iris_z(os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
