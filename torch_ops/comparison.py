import torch

from torch_ops._wrappers import register_bwd_op

def maximum_bwd(dl_dout, mask, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.where(mask, dl_dout, 0, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.where(~mask, dl_dout, 0, out=out[1])
    else:
        dl_dother = None

    return dl_dinput, dl_dother

@register_bwd_op(maximum_bwd)
def maximum(input, other, *, out=None):
    return torch.maximum(input, other, out), (input >= other,) #TODO check if > or >= and how that effects bwd


def minimum_bwd(dl_dout, mask, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.where(mask, dl_dout, 0, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.where(~mask, dl_dout, 0, out=out[1])
    else:
        dl_dother = None

    return dl_dinput, dl_dother

@register_bwd_op(minimum_bwd)
def minimum(input, other, *, out=None):
    return torch.minimum(input, other, out), (input <= other,)


def fmax():
    ...

def fmin():
    ...

def sort_bwd(dl_dout, indicies, dim, shape, *, out=None):
    if out is None:
        out = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    else:
        out = out.zero_()
    return torch.scatter(out, dim, indicies, dl_dout, out=out)

@register_bwd_op(sort_bwd)
def sort(input, dim=-1, descending=False, stable=False, *, out=None):
    sorted, indicies = torch.sort(input, dim, descending, stable, out=out)
    return (sorted, indicies), (indicies, dim, input.shape)


def kthvalue_bwd(dl_dout, indicies, dim, shape, *, out=None):
    if out is None:
        out = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    else:
        out = out.zero_()
    return torch.scatter(out, dim, indicies, dl_dout, out=out)

@register_bwd_op(kthvalue_bwd)
def kthvalue(input, k, dim=None, keepdim=False, *, out=None):
    output, indicies = torch.kthvalue(input, k, dim, keepdim, out=out)
    return (output, indicies), (indicies, dim, input.shape)


def topk_bwd(dl_dout, indicies, dim, shape, *, out=None):
    if out is None:
        out = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    else:
        out = out.zero_()
    return torch.scatter(out, dim, indicies, dl_dout, out=out)

@register_bwd_op(topk_bwd)
def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    out = torch.topk(input, k, dim=dim, largest=largest, sorted=sorted, out=out)
    dim = -1 if dim is None  else dim
    return out , (out.indices, dim, input.shape)

def msort():
    ...