from collections import namedtuple

import torch

from torch_ops._wrappers import register_bwd_op

def argmax(input, dim, keepdim=False):
    return ...

def argmin():
    return ...

#TODO add ability for amax and amin to evenly distributre gradeints between equal values

def amax_bwd(dl_dout, indecies, dim, shape, *, out=None):
    if out is not None:
        dl_din = out.zero_()
    else:
        dl_din = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    return torch.scatter(dl_din, dim, indecies, dl_dout, out=dl_din)

@register_bwd_op(amax_bwd)
def amax(input, dim, keepdim=False, *, out=None):
    indecies = torch.argmax(input, dim, keepdim=keepdim)
    return torch.gather(input, dim, indecies, out=out), (indecies, dim, input.shape)

@register_bwd_op(amax_bwd)
def amin(input, dim, keepdim=False, *, out=None):
    indices = torch.argmin(input, dim, keepdim=keepdim)
    return torch.gather(input, dim, indices, out=out), (indices, dim, input.shape)


def aminmax_bwd(dl_dmin, dl_dmax, minmax_indecies, maxmin_indecies, dim, shape, *, out=None):
    if out is not None:
        dl_din = out.zero_()
    else:
        dl_din = torch.zeros(shape, dtype=dl_dmin.dtype, device=dl_dmin.device)
    dl_din.scatter_(dim, minmax_indecies, dl_dmin)
    dl_din.scatter_add_(dim, maxmin_indecies, dl_dmax)
    return dl_din

@register_bwd_op(aminmax_bwd)
def aminmax(input, *, dim=None, keepdim=False, out=[None, None]):
    #TODO fix keepdim problem
    min_indencies = torch.argmin(input, dim, keepdim=keepdim)
    max_indencies = torch.argmax(input, dim, keepdim=keepdim)
    return namedtuple(min=torch.gather(input, dim, min_indencies, out=out[0]), max=torch.gather(input, dim, max_indencies, out=out[1])), (min_indencies, max_indencies, dim, input.shape)
    

def all():
    ...

def any():
    ...

def max():
    ...

def min():
    ...

def dist():
    ...

def logsumexp():
    ...


def mean_bwd(dl_dout:torch.Tensor, n, input_shape, input_dtype, *, out=None):
    dl_dout = dl_dout.broadcast_to(input_shape) 
    if out is not None:
        return torch.div(dl_dout, n, out=out)
    else:
        return torch.div(dl_dout, n).to(input_dtype)

@register_bwd_op(mean_bwd)
def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dim == None:
        n = input.numel()
        u = torch.mean(input, dtype=dtype)
        if keepdim:
            [1] * input.dim()
            u = u.reshape(input.shape)
        return u, (n, input.shape, input.dtype)
    else:
        if isinstance(dim, int):
            n = input.shape[dim]
        else:
            n = 1
            for i in dim:
                n *= input.shape[i]
        u = torch.mean(input, dim, keepdim=keepdim, dtype=dtype, out=out)
        return u, (n, input.shape, input.dtype)


def nanmean_bwd(dl_dout:torch.Tensor, mask, dim, input_shape, input_dtype, *, out=None):
    dl_dout = dl_dout.broadcast_to(input_shape) 
    if dim is None:
        n = torch.sum(~mask)
    else:
        n = torch.sum(~mask, dim, keepdim=True)
    if out is not None:
        dl_din = torch.div(dl_dout, n, out=out)
    else:
        dl_din = torch.div(dl_dout, n).to(input_dtype)
    dl_din[mask] = 0
    return dl_din
    
@register_bwd_op(nanmean_bwd)
def nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dim == None:
        n = input.numel()
        u = torch.mean(input, dtype=dtype)
        if keepdim:
            [1] * input.dim()
            u = u.reshape(input.shape)
        return u, (torch.isnan(input), dim, input.shape, input.dtype)
    else:
        if isinstance(dim, int):
            n = input.shape[dim]
        else:
            n = 1
            for i in dim:
                n *= input.shape[i]
        u = torch.mean(input, dim, keepdim=keepdim, dtype=dtype, out=out)
        return u, (torch.isnan(input), dim, input.shape, input.dtype)


def median():
    ...

def nanmedian():
    ...


def mode_bwd(dl_dout, indecies, dim, shape, *, out=None):
    if out is not None:
        dl_din = out.zero_()
    else:
        dl_din = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    return torch.scatter(dl_din, dim, indecies, dl_dout, out=dl_din)

@register_bwd_op(mode_bwd)
def mode(input, dim=-1, keepdim=False, *, out=None):
    out = torch.mode(input, dim, keepdim=keepdim, out=out)
    return out, (out.indices, dim, input.shape)


def norm():
    ...

def nansum_bwd(dl_dout, maskdim, dim, input_shape, input_dtype, *, out=None):
    if dim is not None:
        dl_dout = dl_dout.unsqueeze(dim)
    dl_dout = dl_dout.unsqueeze(maskdim).to(input_dtype).broadcast_to(input_shape)
    dl_dout[maskdim] = 0
    if out is not None:
        return out.copy_(dl_dout)
    else:
        return dl_dout

@register_bwd_op(nansum_bwd)
def nansum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dim == None:
        out = torch.nansum(input, dtype=dtype)
        if keepdim:
            broad_shape = [1] * input.dim()
            out = out.reshape(broad_shape)
        return out, (torch.isnan(input), dim, input.shape, input.dtype)
    else:
        return torch.nansum(input, dim, keepdim, dtype, out=out, dtype=dtype), (torch.isnan(input), dim, input.shape, input.dtype)


def prod_bwd(dl_dout, act, dim, dtype, *, out=None):
    dl_dout = dl_dout.unsqueeze(dim).to(dtype).broadcast_to(act.shape)
    return torch.mul(dl_dout, act, out=out)

@register_bwd_op(prod_bwd)
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if dim == None:
        p = torch.prod(input)
        if keepdim:
            broad_shape = [1] * input.dim()
            p = p.reshape(broad_shape)
        return p.to(dtype), (torch.div(p, input), dim, input.dtype)
    else:
        p = torch.prod(input, dim, keepdim=keepdim)
        return p.to(dtype), (torch.div(p.unsqueeze(dim), input), dim, input.dtype)


def quantile():
    ...

def nanquantile():
    ...

def std():
    ...

def std_mean():
    ...


def sum_bwd(dl_dout, dim, input_shape, input_dtype, *, out=None):
    if dim is not None:
        dl_dout = dl_dout.unsqueeze(dim)
    dl_dout = dl_dout.to(input_dtype).broadcast_to(input_shape)
    if out is not None:
        return out.copy_(dl_dout)
    else:
        return dl_dout

@register_bwd_op(sum_bwd)
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if dim == None:
        s = torch.sum(input, dtype=dtype)
        if keepdim:
            broad_shape = [1] * input.dim()
            s = s.reshape(broad_shape)
        return s, (dim, input.shape, input.dtype)
    else:
        s = torch.sum(input, dim, keepdim=keepdim, dtype=dtype)
        return s, (dim, input.shape, input.dtype)

def unique():
    ...

def unique_consecutive():
    ...

def var():
    ...

def var_mean():
    ...
    
def count_nonzero():
    ...

