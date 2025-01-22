import torch
import torch.nn.functional as F

from torch_ops._wrappers import register_bwd_op

def linear_bwd(dl_dout, input, weight, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.matmul(dl_dout.unsqueeze(-2), weight, out=out[0]).squeeze(-2)
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dweight = torch.matmul(input.unsqueeze(-1), dl_dout.unsqueeze(-2), out=out[1])
    else:
        dl_dweight = None
    
    if out_mask[2]:
        dims = dl_dout.shape[:-1]
        dl_dbias = torch.sum(dl_dout, dim=dims, out=out[2])
    else:
        dl_dbias = None
    
    return dl_dinput, dl_dweight, dl_dbias

@register_bwd_op(linear_bwd)
def linear(input, weight, bias=None):
    return F.linear(input, weight, bias), (input, weight)


def bilinear_bwd(dl_dout, input1, input2, weight, *, out_mask=[True, True, True, True], out=[None, None, None, None]):
    if out_mask[0]:
        dl_dinput1 = ...
    else:
        dl_dinput1 = None
    
    if out_mask[1]:
        dl_dinput2 = ...
    else:
        dl_dinput2 = None
    
    if out_mask[2]:
        dl_dweight = ...
    else:
        dl_dweight = None
    
    if out_mask[3]:
        dims = dl_dout.shape[:-2]
        dl_dbias = torch.sum(dl_dout, dim=dims, out=out[3])
    else:
        dl_dbias = None
    
    return dl_dinput1, dl_dinput2, dl_dweight, dl_dbias

@register_bwd_op(bilinear_bwd)
def bilinear(input1, input2, weight, bias=None):
    return F.bilinear(input1, input2, weight, bias), (input1, input2, weight)

