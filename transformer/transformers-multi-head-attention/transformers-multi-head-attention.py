import numpy as np

def multi_head_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
    W_o: np.ndarray, num_heads: int
) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch, n_q, _ = Q.shape
    n_k = K.shape[1]
    n_v = V.shape[1]
    d_model = W_q.shape[1]
    d_k = d_model // num_heads

    Q_h = Q @ W_q
    K_h = K @ W_k
    V_h = V @ W_v

    Q_h = Q_h.reshape(batch, n_q, num_heads, d_k).transpose(0, 2, 1, 3) # (batch, num_heads, n_q, d_k)
    K_h = K_h.reshape(batch, n_k, num_heads, d_k).transpose(0, 2, 1, 3) # (batch, num_heads, n_k, d_k)
    V_h = V_h.reshape(batch, n_v, num_heads, d_k).transpose(0, 2, 1, 3) # (batch, num_heads, n_v, d_k)

    # (batch, num_heads, n_q, d_k) 
    # (batch, num_heads, d_k, n_k)
    scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k) # (batch, num_heads, n_q, n_k)
    scores -= scores.max(axis=-1, keepdims=True) # (batch, num_heads, n_q, n_k)
    weights = np.exp(scores) # (batch, num_heads, d_k, n_k)
    weights /= weights.sum(axis=-1, keepdims=True) # (batch, num_heads, n_q, n_k)

    # (batch, num_heads, n_q, n_k)
    # (batch, num_heads, n_v, d_k) (n_v = n_k) 
    # so V_h (batch, num_heads, n_k, d_k)
    # (batch, num_heads, n_q, d_k)
    attention = weights @ V_h  # (batch, num_heads, n_q, d_k)
    return attention.transpose(0, 2, 1, 3).reshape(batch, n_q, d_model) @ W_o