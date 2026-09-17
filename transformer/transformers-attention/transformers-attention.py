import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    attention_weights = torch.softmax(
        Q @ K.transpose(dim0=-2, dim1=-1) / torch.sqrt(torch.tensor(K.shape[-1], dtype=K.dtype)),
        dim=-1
    )
    return attention_weights @ V