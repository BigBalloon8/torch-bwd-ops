import torch

from torch_ops._wrappers import register_bwd_op

#TODO fuse with @torch.jit.script


def abs_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.sign(input), out=out)

@register_bwd_op(abs_bwd)
def abs(input, *, out=None):
    return torch.abs(input, out=out), (input,)    

@register_bwd_op(abs_bwd)
def absolute(input, *, out=None):
    return torch.abs(input, out=out), (input,)  


def acos_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, (-torch.rsqrt(1-input**2)), out=out)

@register_bwd_op(acos_bwd)
def acos(input, *, out=None) :
    return torch.acos(input, out=out) , (input,)

@register_bwd_op(acos_bwd)
def arccos(input, *, out=None) :
    return torch.acos(input, out=out) , (input,)


def acosh_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.rsqrt(input**2 -1),out=out)

def acosh(input, *, out=None):
    return torch.acosh(input, out=out), (input,)

def arccosh(input, *, out=None):
    return torch.acosh(input, out=out), (input,)


def add_bwd(dl_dout, )

def add(input, other, *, alpha=1, out=None):
    return torch.add(input, other, alpha=alpha, out=out), (input, other, alpha)
    

def addcdiv():
    ...

def addcmul():
    ...

def angle():
    ...

def asin():
    ...

def arcsin():
    ...

def asinh():
    ...

def arcsin():
    ...
    
def atan():
    ...

def arctan():
    ...

def atanh():
    ...

def arctanh():
    ...

def atan2():
    ...

def arctan2():
    ...
    
def bitwise_not():
    ...

def bitwise_and():
    ...
    
def bitwise_or():
    ...

def bitwise_xor():
    ...

def bitwise_left_shift():
    ...
    
def bitwise_right_shift():
    ...

def ceil():
    ...

def clamp():
    ...

def clip():
    ...

def conj_physical():
    ...

def copysign():
    ...

def cos():
    ...

def cosh():
    ...

def deg2rad():
    ...

def div():
    ...

def divide():
    ...

def digamma():
    ...

def erf():
    ...

def erfc():
    ...

def erfinv():
    ...

def exp():
    ...

def exp2():
    ...

def expm1():
    ...

def fix():
    ...
    
def float_power():
    ...

def floor():
    ...

def floor_divide():
    ...

def fmod():
    ...

def frac():
    ...

def frexp():
    ...

def imag():
    ...

def ldexp():
    ...

def lerp():
    ...

def lgamma():
    ...

def log():
    ...

def log10():
    ...

def log1p():
    ...

def log2():
    ...

def logaddexp():
    ...

def logaddexp2():
    ...

def logical_and():
    ...

def logical_not():
    ...

def logical_or():
    ...

def logical_xor():
    ...

def logit():
    ...

def hypot():
    ...

def i0():
    ...

def igamma():
    ...

def igammac():
    ...

def mul():
    ...

def multiply():
    ...

def mvlgamma():
    ...

def nan_to_num():
    ...

def neg():
    ...

def negative():
    ...

def nextafter():
    ...

def polygamma():
    ...

def positive():
    ...

def pow():
    ...

def rad2deg():
    ...

def real():
    ...

def reciprocal():
    ...

def remainder():
    ...

def round():
    ...

def rsqrt():
    ...

def sigmoid():
    ...

def sign():
    ...
    
def sgn():
    ...

def signbit():
    ...

def sin():
    ...

def sinc():
    ...

def sinh():
    ...

def softmax():
    ...

def sqrt():
    ...

def square():
    ...

def sub():
    ...

def substract():
    ...

def tan():
    ...

def tanh():
    ...

def true_divide():
    ...

def trunc():
    ...

def xlogy():
    ...