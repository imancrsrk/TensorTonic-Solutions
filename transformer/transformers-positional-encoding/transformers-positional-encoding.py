import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    res = np.zeros(shape=(seq_length, d_model), dtype=np.float64)
    pos = np.arange(seq_length, dtype=np.float64).reshape(-1, 1)
    i = np.arange(d_model // 2, dtype=np.float64)
    denominator = np.pow(10_000, 2 * i / d_model)
    angles = pos / denominator
    res[:, 0::2] = np.sin(angles)
    res[:, 1::2] = np.cos(angles)
    return res