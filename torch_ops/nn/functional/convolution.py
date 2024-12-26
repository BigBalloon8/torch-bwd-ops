import torch
import torch.nn.functional as F

from torch_ops._wrappers import register_bwd_op


# https://github.com/pytorch/pytorch/blob/3f80632c802f1d9fafd0c303d45ba2376b9c1e11/torch/nn/grad.py#L106

def conv1d_bwd(dl_dout, input, weight, bias, stride, padding, dilation, groups, *, out_mask = [True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.nn.grad.conv1d_input(
            input_size=input.shape,
            weight=weight,
            grad_output=out[0],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[0] is not None:
            dl_dinput = out.copy_(dl_dinput) # theres a better way to do this by calling torch.ops.aten.convolution_backward
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dweight = torch.nn.grad.conv1d_weight(
            input=input,
            weight_size=weight.shape,
            grad_output=out[1],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[1] is not None:
            dl_dweight = out.copy_(dl_dweight)
    else:
        dl_dweight = None

    if out_mask[2] and bias is not None:
        dl_dbias = torch.sum(dl_dout, dim=(0, 2), out=out[2])
    else:
        dl_dbias = None
    return dl_dinput, dl_dweight, dl_dbias

@register_bwd_op(conv1d_bwd)
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1), (input, weight, bias, stride, padding, dilation, groups)


def conv2d_bwd(dl_dout, input, weight, bias, stride, padding, dilation, groups, *, out_mask = [True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.nn.grad.conv2d_input(
            input_size=input.shape,
            weight=weight,
            grad_output=out[0],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[0] is not None:
            dl_dinput = out.copy_(dl_dinput)
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dweight = torch.nn.grad.conv2d_weight(
            input=input,
            weight_size=weight.shape,
            grad_output=out[1],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[1] is not None:
            dl_dweight = out.copy_(dl_dweight)
    else:
        dl_dweight = None
    
    if out_mask[2] and bias is not None:
        dl_dbias = torch.sum(dl_dout, dim=(0, 2, 3), out=out[2])
    else:
        dl_dbias = None
    return dl_dinput, dl_dweight, dl_dbias

@register_bwd_op(conv2d_bwd)
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1), (input, weight, bias, stride, padding, dilation, groups)


def conv3d_bwd(dl_dout, input, weight, bias, stride, padding, dilation, groups, *, out_mask = [True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.nn.grad.conv3d_input(
            input_size=input.shape,
            weight=weight,
            grad_output=out[0],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[0] is not None:
            dl_dinput = out.copy_(dl_dinput)
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dweight = torch.nn.grad.conv3d_weight(
            input=input,
            weight_size=weight.shape,
            grad_output=out[1],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        if out[1] is not None:
            dl_dweight = out.copy_(dl_dweight)
    else:
        dl_dweight = None

    if out_mask[2] and bias is not None:
        dl_dbias = torch.sum(dl_dout, dim=(0, 2, 3, 4), out=out[2])
    else:
        dl_dbias = None
    return dl_dinput, dl_dweight, dl_dbias

@register_bwd_op(conv3d_bwd)
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1), (input, weight, bias, stride, padding, dilation, groups)


def conv_transpose1d():
    ...

def conv_transpose2d():
    ...
    
def conv_transpose3d():
    ...
