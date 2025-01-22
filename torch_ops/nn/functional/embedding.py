import torch

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return torch.nn.functional.embedding(input, weight, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse), (input, weight, scale_grad_by_freq)

def rotary_embeddings(xq, xk, dim, n_heads, seq_len, theta=10000.0):
    """_summary_

    Args:
        xq (torch.Tensor): should be of shape (batch_size, seq_len, dim)
        xk (_type_): _description_
        dim (_type_): _description_
        n_heads (_type_): _description_
        seq_len (_type_): _description_
        theta (float, optional): _description_. Defaults to 10000.0.
    """
    