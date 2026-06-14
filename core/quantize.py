"""Quantization helpers shared across scenarios."""

from __future__ import annotations

import numpy as np


def quantize(vec: np.ndarray, scale: int, dtype=np.int64) -> np.ndarray:
    """Quantize a float vector to integer by multiplying by scale and rounding."""
    return np.round(vec * scale).astype(dtype)


def dequantize(val, scale: int) -> float:
    """Dequantize a scalar score back to float. Divides by scale^2 (dot product of two quantized vecs)."""
    return float(val) / float(scale * scale)
