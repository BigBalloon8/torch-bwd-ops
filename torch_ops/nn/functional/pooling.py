import torch

from torch_ops._wrappers import register_bwd_op

def avg_pool1d():
    return torch.nn.functional.avg_pool1d
    
def avg_pool2d():
    ...

def avg_pool3d():
    ...


def max_pool1d_bwd(dl_dout, indices, kernel_size, stride=None, padding=0, dilation=1, out=None):
    dl_din = torch.nn.functional.max_unpool1d(dl_dout, indices, kernel_size, stride=stride, padding=padding, dilation=dilation)
    if out is not None:
        dl_din = out.copy_(dl_din)
    return dl_din

@register_bwd_op(max_pool1d_bwd)
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1):
    x, indices = torch.nn.functional.max_pool1d(input, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)
    return x, (indices, kernel_size, stride, padding, dilation)


def max_pool2d_bwd(dl_dout, indices, kernel_size, stride=None, padding=0, dilation=1, out=None):
    dl_din = torch.nn.functional.max_unpool2d(dl_dout, indices, kernel_size, stride=stride, padding=padding, dilation=dilation)
    if out is not None:
        dl_din = out.copy_(dl_din)
    return dl_din

@register_bwd_op(max_pool2d_bwd)
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1):
    x, indices = torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)
    return x, (indices, kernel_size, stride, padding, dilation)


def max_pool3d_bwd(dl_dout, indices, kernel_size, stride=None, padding=0, dilation=1, out=None):
    dl_din = torch.nn.functional.max_unpool3d(dl_dout, indices, kernel_size, stride=stride, padding=padding, dilation=dilation)
    if out is not None:
        dl_din = out.copy_(dl_din)
    return dl_din

@register_bwd_op(max_pool3d_bwd)
def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1):
    x, indices = torch.nn.functional.max_pool3d(input, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)
    return x, (indices, kernel_size, stride, padding, dilation)


def max_unpool1d():
    ...
    
def max_unpool2d():
    ...

def max_unpool3d():
    ...


