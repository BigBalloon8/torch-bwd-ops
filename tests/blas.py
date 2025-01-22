import torch

from torch_ops._wrappers import register_bwd_op

def addbmm_bwd(dl_dout, batch1, batch2, beta, alpha, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, beta, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dbatch1 = torch.mul(torch.vmap(torch.mm, in_dims=(None, 0), out_dims=0)(dl_dout, batch2.transpose(-1,-2), out=out[1]), alpha, out=out[1])
    else:
        dl_dbatch1 = None
    
    if out_mask[2]:
        dl_dbatch2 = torch.mul(torch.vmap(torch.mm, in_dims=(0, None), out_dims=1)(batch1.transpose(-1,-2), dl_dout, out=out[2]), alpha, out=out[2])
    else:
        dl_dbatch2 = None
    
    return dl_dinput, dl_dbatch1, dl_dbatch2

@register_bwd_op(addbmm_bwd)
def addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    return torch.addbmm(input, batch1, batch2, beta=beta, alpha=alpha, out=out), (batch1, batch2, beta, alpha)


def addmm_bwd(dl_dout, mat1, mat2, beta, alpha, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, beta, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dmat1 = torch.mul(torch.mm(dl_dout, mat2.transpose(-1,-2), out=out[1]), alpha, out=out[1])
    else:
        dl_dmat1 = None

    if out_mask[2]:
        dl_dmat2 = torch.mul(torch.mm(mat1.transpose(-1,-2), dl_dout, out=out[2]), alpha, out=out[2])
    else:
        dl_dmat2 = None

    return dl_dinput, dl_dmat1, dl_dmat2

@register_bwd_op(addmm_bwd)
def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out), (mat1, mat2, beta, alpha)


def addmv_bwd(dl_dout, mat, vec, beta, alpha, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, beta, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dmat = torch.mul(torch.outer(dl_dout, vec), alpha, out=out[1])
    else:
        dl_dmat = None
    
    if out_mask[2]:
        dl_dvec = torch.mul(torch.mv(mat.transpose(-1,-2), dl_dout.unsqeeuze(1)), alpha, out=out[2]).squeeze(1)
    else:
        dl_dvec = None
    return dl_dinput, dl_dmat, dl_dvec

@register_bwd_op(addmv_bwd)
def addmv(input, mat, vec, *, beta=1, alpha=1, out=None):
    return torch.addmv(input, mat, vec, beta=beta, alpha=alpha, out=out), (mat, vec, beta, alpha)


def addr_bwd(dl_dout, vec1, vec2, beta, alpha, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, beta, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dvec1 = torch.mul(torch.mv(dl_dout, vec2), alpha, out=out[1])
    else:
        dl_dvec1 = None

    if out_mask[2]:
        dl_dvec2 = torch.mul(torch.matmul(vec1.unsqueeze(0), dl_dout).squeeze(0), alpha, out=out[2])
    else:
        dl_dvec2 = None
    return dl_dinput, dl_dvec1, dl_dvec2

@register_bwd_op(addr_bwd)
def addr(input, vec1, vec2, *, beta=1, alpha=1, out=None):
    return torch.addr(input, vec1, vec2, beta=beta, alpha=alpha, out=out), (vec1, vec2, beta, alpha)


def baddbmm_bwd(dl_dout, batch1, batch2, beta, alpha, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, beta, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dbatch1 = torch.mul(torch.bmm(dl_dout, batch2.transpose(-1,-2), out=out[1]), alpha, out=out[1])
    else:
        dl_dbatch1 = None

    if out_mask[2]:
        dl_dbatch2 = torch.mul(torch.bmm(batch1.transpose(-1,-2), dl_dout, out=out[2]), alpha, out=out[2])
    else:
        dl_dbatch2 = None

    return dl_dinput, dl_dbatch1, dl_dbatch2

@register_bwd_op(baddbmm_bwd)
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha, out=out), (batch1, batch2, beta, alpha)


def bmm_bwd(dl_dout, input, mat2, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.bmm(dl_dout, mat2.transpose(-1,-2), out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dmat2 = torch.bmm(input.transpose(-1,-2), dl_dout, out=out[1])
    else:
        dl_dmat2 = None
    return dl_dinput, dl_dmat2

@register_bwd_op(bmm_bwd)
def bmm(input, mat2, *, out=None):
    return torch.bmm(input, mat2, out=out), (input, mat2)


def chain_matmul_bwd(dl_douts, *matrices, mask_out=None, out=None):
    if mask_out is None:
        mask_out = [True]*len(matrices)
    if out is None:
        out = [None]*len(matrices)
    dl_dins = []
    for i in range(len(dl_douts)):
        if not mask_out[i]:
            dl_dins.append(None)
            continue
        if i == 0:
            dl_dins.append(torch.matmul(dl_douts[i], torch.chain_matmul(dl_douts[i+1:]).transpose(-1,-2), out=out[i]))
        elif i == len(dl_douts)-1:
            dl_dins.append(torch.matmul(torch.chain_matmul(dl_douts[:i]).transpose(-1,-2), dl_douts[i], out=out[i]))
        else:
            dl_din = torch.matmul(torch.matmul(torch.chain_matmul(dl_douts[:i]).transpose(-1,-2), dl_douts[i]), torch.chain_matmul(dl_douts[i+1:]).transpose(-1,-2), out=out[i])
            dl_dins.append(dl_din)
    return dl_dins

@register_bwd_op(chain_matmul_bwd)
def chain_matmul(*matrices, out=None):
    return torch.chain_matmul(*matrices, out=out), matrices

def cholesky():
    ...

def cholesky_inverse():
    ...

def cholesky_solve():
    ...


def dot_bwd(dl_dout, input, tensor, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, tensor, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dtensor = torch.mul(dl_dout, input, out=out[1])
    else:
        dl_dtensor = None
    return dl_dinput, dl_dtensor

@register_bwd_op(dot_bwd)
def dot(input, tensor, *, out=None):
    return torch.dot(input, tensor, out=out), (input, tensor)

def geqrf():
    ...


def inner_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, other, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, input, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(inner_bwd)
def inner(input, other, *, out=None):
    return torch.inner(input, other, out=out), (input, other)


def inverse_bwd(dl_dout, inv_t, *, out=None):
    return torch.chain_matmul(inv_t(), dl_dout, inv_t, out=out)

@register_bwd_op(inverse_bwd)
def inverse(input, *, out=None):
    inv = torch.inverse(input, out=out)
    return inv, (inv.transpose(-1,-2),)


def det_bwd(dl_dout, A, det_A, *, out=None):
    return torch.mul(torch.inv(A).transpose(-1,-2), dl_dout*det_A, out=out)

@register_bwd_op(det_bwd)
def det(input):
    det_A = torch.det(input)
    return det_A, (input, det_A)


def logdet():
    ...

def slogdet():
    ...

def lu():
    ...

def lu_solve():
    ...

def lu_unpack():
    ...

def matmul_bwd(dl_dout, input, mat2, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.matmul(dl_dout, mat2.transpose(-1,-2), out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dmat2 = torch.matmul(input.transpose(-1,-2), dl_dout, out=out[1])
    else:
        dl_dmat2 = None
    return dl_dinput, dl_dmat2

@register_bwd_op(matmul_bwd)
def mm(input, mat2, *, out=None):
    return torch.mm(input, mat2, out=out), (input, mat2)
    

def matmul_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    if input.dim() == 1 and other.dim() == 1:
        return dot_bwd(dl_dout, input, other, out_mask=out_mask, out=out)
    elif input.dim() == 1:
        rem_dim = lambda x, y: (x.squeeze(0) , y)
        return rem_dim(matmul_bwd(dl_dout, input.unsqueeze(0), other, out_mask=out_mask, out=out))
    elif other.dim() == 1:
        rem_dim = lambda x, y: (x, y.squeeze(1))
        return rem_dim(matmul_bwd(dl_dout, input, other.unsqueeze(1), out_mask=out_mask, out=out))
    
    if out_mask[0]:
        dl_dinput = torch.matmul(dl_dout, other.transpose(-1,-2), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.matmul(input.transpose(-1,-2), dl_dout, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(matmul_bwd)
def matmul(input, other, *, out=None):
    return torch.matmul(input, other, out=out), (input, other)


def matrix_power_bwd(dl_dout, input, n, *, out=None):
    #TODO verfy this is all correct
    if n == 0:
        return torch.zeros_like(dl_dout, out=out) # I think this is correct
    elif n == 1:
        if out is not None:
            return out.copy_(dl_dout)
        else:
            return dl_dout
    
    if out is None:
        dl_din = torch.zeros_like(dl_dout)
    else: 
        dl_din = out.zero_()
    for i in range(n):
        if i == 0:
            dl_din += torch.matmul(dl_dout, torch.matrix_power(input, n-1).transpose(-1,-2))
        elif i == n-1:
            dl_din += torch.matmul(torch.matrix_power(input, n-1).transpose(-1,-2), dl_dout)
        else:
            dl_din += torch.matmul(torch.matmul(torch.matrix_power(input, i).transpose(-1,-2), dl_dout), torch.matrix_power(input, n-1-i).transpose(-1,-2)) # not 100% sure about this

@register_bwd_op(matrix_power_bwd)
def matrix_power(input, n, *, out=None):
    return torch.matrix_power(input, n, out=out), (input, n)


def matrix_exp():
    ...


def mv_bwd(dl_dout, mat, vec, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dmat = torch.outer(dl_dout, vec, out=out[0])
    else:
        dl_dmat = None

    if out_mask[1]:
        dl_dvec = torch.mv(mat.transpose(-1,-2), dl_dout, out=out[1])
    else:
        dl_dvec = None
    return dl_dmat, dl_dvec

@register_bwd_op(mv_bwd)
def mv(input, vec, *, out=None):
    return torch.mv(input, vec, out=out), (input, vec)


def orgqr():
    ...

def ormqr():
    ...

def outer_bwd(dl_dout, input, vec2, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mv(dl_dout, vec2, out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dvec2 = torch.mv(dl_dout, input, out=out[1])
    else:
        dl_dvec2 = None
    return dl_dinput, dl_dvec2

@register_bwd_op(outer_bwd)
def outer(input, vec2, *, out=None):
    return torch.outer(input, vec2, out=out), (input, vec2)

@register_bwd_op(outer_bwd)
def ger(input, vec2, *, out=None):
    return torch.ger(input, vec2, out=out), (input, vec2)
    
def pinverse():
    ...

def qr():
    ...

def svd():
    ...

def svd_lowrank():
    ...

def pca_lowrank():
    ...

def lobpcg():
    ...

def trapz():
    ...

def trapezoid():
    ...

def cumulative_trapezoid():
    ...

def triangular_solve():
    ...

def vdot():
    ...
