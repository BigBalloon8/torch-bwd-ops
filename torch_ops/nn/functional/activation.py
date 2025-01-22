import torch
import torch.nn.functional as F

from torch_ops._wrappers import register_bwd_op


# TODO inplace operation for bwd could be implemented by using the masked_fill_ 

def threshold_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(threshold_bwd)
def threshold(input, threshold, value, inplace=False):
    return F.threshold(input, threshold, value, inplace=inplace), (input>threshold,)

@register_bwd_op(threshold_bwd)
def threshold_(input, threshold, value):
    return F.threshold(input, threshold, value, inplace=True), (input>threshold,)


def relu_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(relu_bwd)
def relu(input, inplace=False):
    return F.relu(input, inplace=inplace), (input>0,)

@register_bwd_op(relu_bwd)
def relu_(input):
    return F.relu(input, inplace=True), (input>0,)


def hardtanh_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(hardtanh_bwd)
def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    return F.hardtanh(input, min_val, max_val, inplace=inplace), (torch.bitwise_or(input>min_val, input<max_val),)

@register_bwd_op(hardtanh_bwd)
def hardtanh_(input, min_val=-1, max_val=1):
    return F.hardtanh(input, min_val, max_val, inplace=True), (torch.bitwise_or(input>min_val, input<max_val),)


def hardswish_bwd(dl_dout, input, out=None):
    inter_d = torch.where(input<-3,0,dl_dout)
    return torch.where(input>3, inter_d, inter_d*(2*input/6-0.5), out=out)

@register_bwd_op(hardswish_bwd)
def hardswish(input, inplace=False):
    return F.hardswish(input, inplace=inplace), (input)


def relu6_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(relu6_bwd)
def relu6(input, inplace=False):
    return F.relu6(input, inplace=inplace), (torch.bitwise_and(input>0, input<6),)


def elu_bwd(dl_dout, input, alpha, out=None):
    return torch.where(input>0, dl_dout, alpha*torch.exp(input)*dl_dout, out=out)

@register_bwd_op
def elu(input, alpha=1.0, inplace=False):
    if inplace:
        raise NotImplementedError("inplace is not supported for elu for backward pass efficency")
    return torch.where(input>0, input, alpha*(torch.exp(input)-1)) (input, alpha)


def elu_(*args, **kwrags):
    raise NotImplementedError("inplace is not supported for elu for backward pass efficency")


def selu_bwd(dl_dout, input):
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale*torch.where(input>0, dl_dout, alpha*torch.exp(input)*dl_dout)

@register_bwd_op(selu_bwd)
def selu(input, inplace=False):
    # This implementation could be wrong
    if inplace:
        raise NotImplementedError("inplace is not supported for selu for backward pass efficency")
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale*torch.where(input>0, input, alpha*(torch.exp(input)-1)), (input,)


@register_bwd_op(elu_bwd)
def celu(input, alpha, inplace):
    return F.celu(input, alpha, inplace=inplace), (input, alpha)
    

def leaky_relu_bwd(dl_dout, mask, negative_slope, out=None):
    return torch.where(mask, dl_dout, negative_slope*dl_dout, out=out)

@register_bwd_op(leaky_relu_bwd)
def leaky_relu(input, negative_slope=0.01, inplace=False):
    mask = input>0
    return torch.where(mask, input, negative_slope*input), (mask, negative_slope)

@register_bwd_op(leaky_relu_bwd)
def leaky_relu_(*args, **kwrags):
    raise NotImplementedError("inplace is not supported for leaky_relu for backward pass efficency")


def prelu_bwd(dl_dout, input, weight, out=None):
    return torch.where(input>0, dl_dout, weight*dl_dout, out=out)

@register_bwd_op(prelu_bwd)
def prelu(input, weight):
    return F.prelu(input, weight), (input>0, weight)


def rrelu_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(rrelu_bwd)
def rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False):
    return F.rrelu(input, lower, upper, training, inplace), (input>0,)


def glu_bwd(dl_dout, a, sigmoid_b, dim, *, out=None):
    return torch.cat((dl_dout*sigmoid_b, dl_dout*a(sigmoid_b*(1-sigmoid_b))), dim=dim, out=out)

@register_bwd_op(glu_bwd)
def glu(input, dim=-1):
    a,b = torch.chunk(input, 2, dim=dim)
    sigmoid_b = torch.sigmoid(b)
    return a * sigmoid_b, (a, sigmoid_b, dim)


def gelu_bwd(dl_dout, input, out=None):
    return torch.mul(dl_dout, (0.5 * (1 + torch.erf(input/((2)**(0.5)))) + (input)/torch.pi**0.5 * torch.exp(-torch.pow(input, 2))), out=out)

@register_bwd_op(gelu_bwd)
def gelu(input, approximation="none"):
    return torch.nn.functional.gelu(input, approximation=approximation), (input,)


def logsigmoid_bwd(dl_dout, sig_in, out=None):
    return torch.mul(dl_dout, (1-sig_in), out=out)

@register_bwd_op(logsigmoid_bwd)
def logsigmoid(input):
    sig = torch.sigmoid(input)
    return torch.log(sig), (sig,)


def hardshrink_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(hardshrink_bwd)
def hardshrink(input, lambd=0.5):
    mask = torch.bitwise_and(input>lambd, input<-lambd)
    return torch.where(mask, input, 0), (mask,)


def tanhshrink_bwd(dl_dout, tanh_x, out=None):
    return torch.mul(dl_dout, tanh_x**2, out=out)

@register_bwd_op(tanhshrink_bwd)
def tanhshrink(input):
    tanh_x = torch.tanh(input)
    return input-tanh_x, (tanh_x,)


def softsign_bwd(dl_dout, input, out=None):
    return torch.mul(dl_dout, 1/((1+torch.abs(input))**2), out=out)

@register_bwd_op(softsign_bwd)
def softsign(input):
    return F.softsign(input), (input,)


def softplus_bwd(dl_dout, input, beta, threshold, out=None):
    exp_x = torch.exp(input*beta)
    return torch.where(input<threshold, dl_dout*(exp_x/(1+exp_x)), dl_dout), (input, beta)

@register_bwd_op(softplus_bwd)
def softplus(input, beta=1, threshold=20):
    return F.softplus(input, beta, threshold), (input, beta)


def softmin():
    ...


def softmax_bwd(dl_dout, s, dim, dtype, out=None):
    # i think this is probably incorrect as softmax subtracts the max from the input
    m = s*dl_dout
    if dtype is None:
        return torch.sub(m, s*torch.sum(m, dim=dim, keepdim=True), out=out)
    else:
        dl_din = torch.sub(m, s*torch.sum(m, dim=dim, keepdim=True))
        if out is not None:
            return out.copy_(dl_din.to(dtype))
        else:
            return dl_din.to(dtype)

@register_bwd_op(softmax_bwd)
def softmax(input, dim=None, dtype=None):
    out = torch.softmax(input, dim, dtype=dtype)
    return out, (out, dim, dtype)


def softshrink_bwd(dl_dout, mask, out=None):
    return torch.where(mask, dl_dout, 0, out=out)

@register_bwd_op(softshrink_bwd)
def softshrink(input, lambd=0.5):
    mask1 = input>lambd
    mask2 = input<-lambd
    return torch.where(mask1, input-lambd, torch.where(mask2, input+lambd, 0)), (torch.bitwise_and(mask1, mask2),)


def gumbel_softmax():
    ...


def log_softmax_bwd(dl_dout, s, out=None):
    return torch.mul(dl_dout, s-1, out=out)

@register_bwd_op(log_softmax_bwd)
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    s = torch.softmax(input, dim, _stacklevel, dtype)
    return torch.log(s), (s,)


def tanh_bwd(dl_dout, tanh_x, out=None):
    return torch.mul(dl_dout, 1-tanh_x**2, out=out)

@register_bwd_op(tanh_bwd)
def tanh(input):
    tanh_x =  torch.tanh(input)
    return tanh_x, (tanh_x,)


def sigmoid_bwd(dl_dout, sig_x, out=None):
    return torch.mul(dl_dout, sig_x*(1-sig_x), out=out)

@register_bwd_op(sigmoid_bwd)
def sigmoid(input):
    s_x = torch.sigmoid(input)
    return s_x, (s_x,)


def hardsigmoid_bwd(dl_dout, mask, out=None):
    return torch.where(mask, 0, (1/6)*dl_dout, out=out)

@register_bwd_op(hardsigmoid_bwd)
def hardsigmoid(input, inplace=False):
    return F.hardsigmoid(input, inplace=inplace), (torch.bitwise_and(input>=3, input<=-3),)
    

def silu_bwd(dl_dout, input, out=None):
    e_x = torch.exp(-input)
    return torch.mul(dl_dout, ((1+e_x) + input*e_x)/((1+e_x)**2), out=out)

@register_bwd_op(silu_bwd)
def silu(input, inplace=False):
    return F.silu(input, inplace=inplace), (input,)

def mish():
    ...

def batch_norm():
    ...

def group_norm():
    ...

def instance_norm():
    ...

def layer_norm():
    ...

def local_response_norm():
    ...

def normalize():
    ...

