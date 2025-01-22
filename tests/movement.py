import torch

from torch_ops._wrappers import register_bwd_op


def cat_bwd(dl_dout, dim, sizes):
    return torch.split(dl_dout, sizes, dim=dim)

@register_bwd_op(cat_bwd)
def cat(tensors, dim=0, *, out=None):
    sizes = [t.size(dim) for t in tensors]
    return torch.cat(tensors, dim=dim, out=out), (dim, sizes)

@register_bwd_op(cat_bwd)
def concat(tensors, dim=0, *, out=None):
    sizes = [t.size(dim) for t in tensors]
    return torch.cat(tensors, dim=dim, out=out), (dim, sizes)


def chunk_bwd(dl_douts, dim, *, out=None):
    return torch.cat(dl_douts, dim=dim, out=out)

@register_bwd_op(chunk_bwd)
def chunk(input, chunks, dim=0):
    return torch.chunk(input, chunks, dim=dim), (dim,)


def dsplit(input, indices_or_sections):
    #TODO implement once tensor_split is implemented as eqiv
    ...

def column_stack():
    ...

def dstack():
    ...

def gather_backward(dl_dout, dim, index, shape, *, out=None):
    # TODO test whether correct
    if out is None:
        out = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    return out.scatter_(dim, index, dl_dout)

@register_bwd_op(gather_backward)
def gather(input, dim, index, *, out=None):
    return torch.gather(input, dim, index, out=out), (dim, index, input.shape)


def hsplit():
    ...

def hstack():
    ...


def index_add_bwd(dl_dout, dim, index, alpha, *, out_mask = [True, True], out=[None,None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].clone_(dl_dout)
        else:
            dl_dinput = dl_dinput
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dsource = alpha*torch.index_select(dl_dout, dim, index, out=out[1])
    else:
        dl_dsource = None
    return dl_dinput, dl_dsource

@register_bwd_op(index_add_bwd)
def index_add(input, dim, index, source, *, alpha=1, out=None):
    if out is None:
        out = input.clone()
    else:
        out = out.copy_(input)
    return out.index_add_(dim, index, source, alpha=alpha), (dim, index, alpha)


def index_copy_bwd(dl_dout, dim, index, *, out_mask = [True, True], out=[None,None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].clone_(dl_dout)
        else:
            dl_dinput = dl_dout
        dl_dinput.index_fill_(dim, index, 0)
    else:
        dl_dinput = None
    if out_mask[1]:
        if out[1] is not None:
            dl_dsource = out[1].clone_(dl_dout)
        else:
            dl_dsource = dl_dout
        # TODO I think index can only have dim 1 need to check
        inv_indexs = torch.tensor([i for i in range(dl_dout.size(dim)) if i not in torch.asarray(index)])
        dl_dsource.index_fill_(dim, inv_indexs, 0)
    else:
        dl_dsource = None
    return dl_dinput, dl_dsource

@register_bwd_op(index_copy_bwd)
def index_copy(input, dim, index, source, *, out=None):
    if out is None:
        out = input.clone()
    else:
        out = out.copy_(input)
    return out.index_copy_(dim, index, source), (dim, index)


def index_reduce():
    ...


def index_select_bwd(dl_dout, dim, index, shahpe, *, out=None):
    if out is None:
        out = torch.zeros(shahpe, dtype=dl_dout.shape, device=dl_dout.device)
    else:
        out = out.zero_()
    return torch.index_copy(out, dim, index, dl_dout)
    
@register_bwd_op(index_select_bwd)
def index_select(input, dim, index, *, out=None):
    return torch.index_select(input, dim, index, out=out), (dim, index, input.shape)


def masked_fill_bwd(dl_dout, mask, shape, dtype, *, out=None):
    if out is not None:
        dl_dinput = out.zero_().flatten()
    else:
        dl_dinput = torch.zeros(shape, dtype=dtype, device=dl_dout.device).flatten()
    dl_dinput[mask] = dl_dout.flatten()
    return dl_dinput.reshape(shape)

@register_bwd_op(masked_fill_bwd)
def masked_select(input, mask, *, out=None):
    return torch.masked_select(input, mask, out=out), (mask, input.shape, input.dtype)


def movedim():
    ...

def moveaxis():
    ...


def narrow_bwd(dl_dout, dim, start, length, shape, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    torch.narrow(dl_dinput, dim, start, length).copy_(dl_dout)
    return dl_dinput

@register_bwd_op(narrow_bwd)
def narrow(input, dim, start, length):
    return torch.narrow(input, dim, start, length), (dim, start, length, input.shape)

@register_bwd_op(narrow_bwd)
def narrow_copy(input, dim, start, length, *, out=None):
    return torch.narrow_copy(input, dim, start, length, out=out), (dim, start, length, input.shape)


def nonzero():
    ...


def permute_bwd(dl_dout, dims):
    return torch.permute(dl_dout, dims)

@register_bwd_op(permute_bwd)
def permute(input, dims):
    return torch.permute(input, dims), (dims,)


def reshape_backward(dl_dout, shape,):
    return dl_dout.reshape(shape)

@register_bwd_op(reshape_backward)
def reshape(input, shape):
    return torch.reshape(input, shape), (input.shape,)


def row_stack():
    ...


def select_bwd(dl_dout, dim, index, shape, *, out=None):
    if out is None:
        dl_din = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    else:
        dl_din = out.zero_()
    vec = torch.select(dl_din, dim, index)
    vec.copy_(dl_dout)
    return dl_din

@register_bwd_op(select_bwd)
def select(input, dim, index):
    return torch.select(input, dim, index), (dim, index, input.shape)


def scatter():
    ...

def diagonal_scatter():
    ...

def select_scatter():
    ...

def slice_scatter():
    ...

def scatter_add():
    ...

def scatter_reduce():
    ...


def split_bwd(dl_douts, dim, *, out=None):
    return torch.cat(dl_douts, dim=dim, out=out)

@register_bwd_op(split_bwd)
def split(tensor, split_size_or_sections, dim=0):
    return torch.split(tensor, split_size_or_sections, dim=dim), (dim,)


def squeeze_bwd(dl_dout, shape):
    return dl_dout.reshape(shape)

@register_bwd_op(squeeze_bwd)
def squeeze(input, dim):
    return torch.squeeze(input, dim), (input.shape)


def stack_bwd(dl_dout, dim):
    return [torch.squeeze(x, dim) for x in torch.chunk(dl_dout, dl_dout.size(dim), dim)]

@register_bwd_op(stack_bwd)
def stack(tensors, dim=0, *, out=None):
    return torch.stack(tensors, dim=dim, out=out), (dim,)


def t_bwd(dl_dout):
    return dl_dout.t

@register_bwd_op(t_bwd)
def t(input):
    return torch.t(input)


def take_bwd(dl_dout, index, shape, *,out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    flat_storage = dl_dinput.flatten()
    flat_storage[index] = dl_dout.flatten()
    return dl_dinput

@register_bwd_op(take_bwd)
def take(input, index):
    return torch.take(input, index), (index, input.shape)


def take_along_dim_bwd(dl_dout, indices, dim, shape, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros(shape, dtype=dl_dout.dtype, device=dl_dout.device)
    return torch.scatter(dl_dinput, dim, indices, dl_dout) # TODO check, might be wrong

@register_bwd_op(take_along_dim_bwd)
def take_along_dim(input, indices, dim=None, *, out=None):
    return torch.take_along_dim(input, indices, dim=dim, out=out), (indices, dim, input.shape)
    

def tensor_split():
    ...


def tile_bwd(dl_dout, dims, *, out=None):
    ...

def tile(input, dims):
    return torch.tile(input, dims), (dims,)


def transpose_bwd(dl_dout, dim0, dim1):
    return torch.transpose(dl_dout, dim0, dim1)

@register_bwd_op(transpose_bwd)
def transpose(input, dim0, dim1):
    return torch.transpose(input, dim0, dim1), (dim0, dim1)

@register_bwd_op(transpose_bwd)
def swapaxes(input, axis0, axis1):
    return torch.swapaxes(input, axis0, axis0), (axis0, axis1)

@register_bwd_op(transpose_bwd)
def swapdims(input, dim0, dim1):
    return torch.swapdims(input, dim0, dim0), (dim0, dim1)


def unbind_backward(dl_douts, dim, *, out=None):
    return torch.cat(dl_douts, dim=dim, out=out)

@register_bwd_op(unbind_backward)
def unbind(input, dim=0):
    return torch.unbind(input, dim=dim), (dim)


def unravel_index():
    ... # doesnt have a backward


def unsqueeze_bwd(dl_dout, dim):
    return dl_dout.squeeze(dim)

def unsqeeze(input, dim):
    return torch.unsqueeze(input, dim), (dim,)

def vsplit():
    ...

def vstack():
    ...

def where_bwd(dl_dout, condition, *, out_mask= [True, True], out=[None,None]):
    if out_mask[0]:
        dl_dinput = torch.where(condition, dl_dout, 0, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.where(condition, 0, dl_dout, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

def where(condition, input, other, *, out=None):
    return torch.where(condition, input, other, out=out), (condition)

