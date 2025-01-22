import torch

from torch_ops._wrappers import register_bwd_op

def dropout_bwd(dl_dout, mask, out=None):
    if isinstance(mask, bool):
        if mask:
            return dl_dout
        else:
            return torch.zeros_like(dl_dout)
    else:
        return dl_dout*mask

@register_bwd_op(dropout_bwd)
def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p==0:
        return input, (True,)
    if p==1:
        return torch.zeros_like(input), (False)
    with torch.no_grad():
        p_mask = torch.bernoulli(torch.ones_like(input) - p)
    return input*p_mask, (p_mask,) # masked fill may be faster

def alpha_dropout():
    ...

def feature_alpha_dropout():
    ...

def dropout_channel_bwd(dl_dout, mask, out=None):
    if isinstance(mask, bool):
        if mask:
            return dl_dout
        else:
            return torch.zeros_like(dl_dout)
    else:
        dl_dout = dl_dout[mask] = 0
        return dl_dout

@register_bwd_op(dropout_channel_bwd)
def dropout_channel(input, p=0.5, training=True, inplace=False):
    if not training or p==0:
        return input, (True,)
    if p==1:
        return torch.zeros_like(input), (False)
    with torch.no_grad():
        p_mask = torch.bernoulli(torch.full(input.shape[:2], fill_value=p, device=input.device))
        if not inplace:
            input = input.clone()
        input[p_mask.to(torch.bool)] = 0
    return input, (p_mask,) # masked fill may be faster

dropout1d = dropout_channel

dropout2d = dropout_channel
    
dropout3d = dropout_channel