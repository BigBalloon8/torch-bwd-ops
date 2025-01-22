import math

import torch

from torch_ops._wrappers import register_bwd_op

#TODO fuse with @torch.jit.script
#TODO add unbroadcast support


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

@register_bwd_op(acos_bwd)
def acosh(input, *, out=None):
    return torch.acosh(input, out=out), (input,)

@register_bwd_op(acos_bwd)
def arccosh(input, *, out=None):
    return torch.acosh(input, out=out), (input,)


def add_bwd(dl_dout, alpha, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].copy_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        if out[1] is not None:
            dl_dother = out[1].copy_(dl_dout if alpha == 1 else alpha*dl_dout)
        else:
            dl_dout if alpha == 1 else alpha*dl_dout
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(add_bwd)
def add(input, other, *, alpha=1, out=None):
    return torch.add(input, other, alpha=alpha, out=out), (alpha,)
    

def addcdiv_bwd(dl_dout, tensor1, tensor2_inv, value, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].copy_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dtensor1 = torch.mul(value*dl_dout, tensor2_inv, out=out[1])
    else:
        dl_dtensor1 = None
    
    if out_mask[2]:        
        dl_dtensor2 = torch.mul(-value*dl_dout, tensor2_inv**2, out=out[2])
    return dl_dinput, dl_dtensor1, dl_dtensor2

@register_bwd_op(addcdiv_bwd)
def addcdiv(input, tensor1, tensor2, *, value=1, out=None):
    t2_inv = torch.reciprocal(tensor2)
    return torch.addcmul(input, tensor1, t2_inv, value=value, out=out), (tensor1, t2_inv, value)


def addcmul_bwd(dl_dout, tensor1, tensor2, value, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].copy_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        #TODO could reuse value*dl_dout
        dl_dtensor1 = torch.mul(value*dl_dout, tensor2, out=out[1])
    else:
        dl_dtensor1 = None
    
    if out_mask[2]:
        dl_dtensor2 = torch.mul(value*dl_dout, tensor1, out=out[2])
    else:
        dl_dtensor2 = None
    return dl_dinput, dl_dtensor1, dl_dtensor2
        
@register_bwd_op(addcmul_bwd)
def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    return torch.addcmul(input, tensor1, tensor2, value=value, out=out), (tensor1, tensor2, value)
    

def angle():
    ...


def asin_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, (torch.rsqrt(1-input**2)), out=out)

@register_bwd_op(asin_bwd)
def asin(input, *, out=None):
    return torch.asin(input, out=out), (input,)

@register_bwd_op(asin_bwd)
def arcsin(input, *, out=None):
    return torch.asin(input, out=out), (input,)


def asinh_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.rsqrt(input**2 + 1), out=out)

@register_bwd_op(asinh_bwd)
def asinh(input, *, out=None):
    return torch.asinh(input, out=out), (input,)

@register_bwd_op(asinh_bwd)
def arcsin(input, *, out=None):
    return torch.asinh(input, out=out), (input,)


def atan_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, 1+input**2, out=out)

@register_bwd_op(atan_bwd)
def atan(input, *, out=None):
    return torch.atan(input, out=out), (input,)

@register_bwd_op(atan_bwd)
def arctan(input, *, out=None):
    return torch.atan(input, out=out), (input,)


def atanh_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, 1-input**2, out=out)

@register_bwd_op(atanh_bwd)
def atanh(input, *, out=None):
    return torch.atanh(input, out=out), (input,)

@register_bwd_op(atanh_bwd)
def arctanh(input, *, out=None):
    return torch.atanh(input, out=out), (input,)


def atan2_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    #TODO reuse input**2 and other**2
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, torch.reciprocal(1+torch.div(input, other)**2), out=out[0])
    else:
        dl_dinput = None

    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, -torch.reciprocal(input**2 + other**2), out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother
        
@register_bwd_op(atan2_bwd)
def atan2(input, other, *, out=None):
    return torch.atan2(input, other, out=out), (input, other)

@register_bwd_op(atan2_bwd)
def arctan2(input, other, *, out=None):
    return torch.atan2(input, other, out=out), (input, other)


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


def ceil_bwd(dl_dout, _, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(ceil_bwd)
def ceil(input, *, out=None):
    return torch.ceil(input, out=out), (None,) 


def clamp_bwd(dl_dout, input, min, max, *, out_mask=[True, True, True], out=[None, None, None]):
    if min is not None:
        mask_l = input < min
    if max is not None:
        mask_u = input > max
    
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout,  torch.bitwise_not(mask_l + mask_u), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1] and isinstance(min, torch.Tensor):
        dl_dmin = torch.mul(dl_dout, mask_l, out=out[1])
    else:
        dl_dmin = None
    
    if out_mask[2] and isinstance(max, torch.Tensor):
        dl_dmax = torch.mul(dl_dout, mask_u, out=out[2])
    else:
        dl_dmax = None
    return dl_dinput, dl_dmin, dl_dmax
            
@register_bwd_op(clamp_bwd)
def clamp(input, min=None, max=None, *, out=None):
    return torch.clamp(input, min, max, out=out), (input, min, max)

@register_bwd_op(clamp_bwd)
def clip(input, min=None, max=None, *, out=None):
    return torch.clamp(input, min, max, out=out), (input, min, max)


def conj_physical_bwd(dl_dout, _, *, out=None):
    if not input.is_complex():
        return dl_dout
    else:
        if out is not None:
            out.real, out.imag = dl_dout.real, -dl_dout.imag
        else:
            out = torch.empty_like(dl_dout)
            out.real, out.imag = dl_dout.real, -dl_dout.imag
        return out

@register_bwd_op(conj_physical_bwd)
def conj_physical(input, *, out=None):
    return torch.conj_physical(input, out=out), (None,)


def copysign_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        neg_mask = torch.logical_xor(torch.signbit(input), torch.signbit(other))
        dl_dinput = torch.where(neg_mask, -dl_dout, dl_dout, out=out)
    if out_mask[1]:
        if out[1] is not None:
            dl_dother = out.copy_(dl_dout)
        else:
            dl_dother = dl_dout
    return dl_dinput, dl_dother

@register_bwd_op(copysign_bwd)
def copysign(input, other, *, out=None):
    return torch.copysign(input, other, out=out), (input, other,)


def cos_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, -torch.sin(input), out=out)

@register_bwd_op(cos_bwd)
def cos(input, *, out=None):
    return torch.cos(input, out=out), (input,)


def cosh_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.sinh(input), out=out)

def cosh(input, *, out=None):
    return torch.cosh(input, out=out), (input,)


def deg2rad_bwd(dl_dout, _, *, out=None):
    return torch.mul(dl_dout, torch.pi/180, out=out)

@register_bwd_op(deg2rad_bwd)
def deg2rad(input, *, out=None):
    return torch.deg2rad(input, out=out), (None,)


def div_bwd(dl_dout, input, other, rounding_mode, *, out_mask=[True, True], out=[None, None]):
    #TODO potentially reuse 1/other 
    if rounding_mode in ("trunc", "floor"):
        if out_mask[0]:
            if out[0] is not None:
                dl_dinput = out[0].zero_()
            else:
                dl_dinput = torch.zeros_like(dl_dout)
        else:
            dl_dinput = None
        if out_mask[1]:
            if out[1] is not None:
                dl_dother = out[1].zero_()
            else:
                dl_dother = torch.zeros_like(dl_dout)
        else:
            dl_dother = None
        return dl_dinput, dl_dother
    
    if out_mask[0]:
        dl_dinput = torch.div(dl_dout, other, out=out[0])
    else:
        dl_dinput = None
    if out_mask[1]:
        dl_dother = torch.div(-dl_dout*input, other**2, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(div_bwd)
def div(input, other, *, rounding_mode=None, out=None):
     return torch.div(input, other, rounding_mode=rounding_mode, out=out), (input, other, rounding_mode)

@register_bwd_op(div_bwd)
def divide(input, other, *, rounding_mode=None, out=None):
     return torch.div(input, other, rounding_mode=rounding_mode, out=out), (input, other, rounding_mode)


def digamma():
    ...


def erf_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, (2/math.sqrt(torch.pi))*torch.exp(-input**2), out=out)

@register_bwd_op(erf_bwd)
def erf(input, *, out=None):
    return torch.erf(input, out=out), (input,)


def erfc_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, -(2/math.sqrt(torch.pi))*torch.exp(-input**2), out=out)

@register_bwd_op(erfc_bwd)
def erfc(input, *, out=None):
    return torch.erfc(input, out=out), (input,)


def erfinv_bwd(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, (2/math.sqrt(torch.pi))*torch.exp(-y**2), out=out)

@register_bwd_op(erfinv_bwd)
def erfinv(input, *, out=None):
    y = torch.erfinv(input, out=out)
    return y , (y,)


def exp_bwd(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, y, out=out)

@register_bwd_op(exp_bwd)
def exp(input, *, out=None):
    y = torch.exp(input, out=out)
    return y, (y,)


def exp2_bwd(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, math.log(2)*y, out=out)

@register_bwd_op(exp2_bwd)
def exp2(input, *, out=None):
    y = torch.exp2(input, out=out)
    return y , (y,)


def expm1_bwd(dl_dout, ex_m1, *, out=None):
    return torch.addcmul(dl_dout, tensor1=dl_dout, tensor2=ex_m1)
    return torch.mul(dl_dout, e_x+1, out=out)

@register_bwd_op(expm1_bwd)
def expm1(input, *, out=None):
    ex_m1 = torch.expm1(input, out=out)
    return ex_m1, (ex_m1,)


def float_power_bwd(dl_dout, input, exponent, y, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, exponent* torch.float_power(input, exponent-1), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dexponent = torch.mul(dl_dout, torch.log(input)*y, out=out[1])
    else:
        dl_dexponent = None
    return dl_dinput, dl_dexponent

@register_bwd_op(float_power_bwd)
def float_power(input, exponent, *, out=None):
    y = torch.float_power(input, exponent, out=out)
    return y, (input, exponent, y)


def floor_bwd(dl_dout, _, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(floor_bwd)
def floor(input, *, out=None):
    return torch.floor(input, out=out), (None,)
    

def floor_divide_bwd(dl_dout, _, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].zero_()
        else:
            dl_dinput = torch.zeros_like(dl_dout)
    else:
        dl_dinput = None
    if out_mask[1]:
        if out[1] is not None:
            dl_dother = out[1].zero_()
        else:
            dl_dother = torch.zeros_like(dl_dout)
    else:
        dl_dother = None
    return dl_dinput, dl_dother
        
@register_bwd_op(floor_divide_bwd)
def floor_divide(input, other, *, out=None):
    return torch.floor_divide(input, other, out=out), (None,)


def fmod_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None,None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].clone_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, -input.div(other, rounding_mode="trunc"), out=out)
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(fmod_bwd)
def fmod(input, other, *, out=None):
    return torch.fmod(input, other, out=out), (input, other,)


def frac_bwd(dl_dout, y, *, out=None):
    return torch.where(y==0, 0, dl_dout, out=out)

@register_bwd_op(frac_bwd)
def frac(input, *, out=None):
    y = torch.frac(input, out=out)
    return y, (y,)


def frexp():
    ...


def ldexp(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    other_2 =  torch.exp2(other)
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, other_2, out=out[0])
    else:
        dl_dinput = None
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, input*math.log(2)*other_2, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

def ldexp(input, other, *, out=None):
    return torch.ldexp(input, other, out=out), (input, other)
    
    
def lerp_bwd(dl_dout, input, end, weight, *, out_mask=[True, True, True], out=[None, None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, 1-weight, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dend = torch.mul(dl_dout, weight, out=out[1])
    else:
        dl_dend = None
        
    if out_mask[2]:
        dl_dweight = torch.mul(dl_dout, end-input, out=out[2])
    else:
        dl_dweight = None
    return dl_dinput, dl_dend, dl_dweight

@register_bwd_op(lerp_bwd)
def lerp(input, end, weight, *, out=None):
    return torch.lerp(input, end, weight, out=out), (input, end, weight)


def lgamma():
    ...


def log_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, input, out=out)

@register_bwd_op(log_bwd)
def log(input, *, out=None):
    return torch.log(input, out=out), (input,)


def log10_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, math.log(10)*input, out=out)

@register_bwd_op(log10_bwd)
def log10(input, *, out=None):
    return torch.log10(input, out=out), (input,)


def log1p_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, 1+input, out=out)

@register_bwd_op(log1p_bwd)
def log1p(input, *, out=None):
    return torch.log1p(input, out=out), (input,)


def log2_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, math.log(2)*input, out=out)

@register_bwd_op(log2_bwd)
def log2(input, *, out=None):
    return torch.log2(input, out=out), (input,)


def logaddexp_bwd(dl_dout, e_x, e_y, out_mask=[True, True], out=[None, None]):
    #TODO reuse e_x + e_y
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, e_x/(e_x + e_y), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, e_y/(e_x + e_y), out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(logaddexp_bwd)
def logaddexp(input, other, *, out=None):
    e_x, e_y = torch.exp(input), torch.exp(other)
    return torch.log(e_x+e_y, out=out), (e_x, e_y)


def logaddexp2_bwd(dl_dout, _2_x, _2_y, out_mask=[True, True], out=[None, None]):
    #TODO reuse e_x + e_y
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, _2_x/(_2_x + _2_y), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, _2_y/(_2_x + _2_y), out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(logaddexp2_bwd)
def logaddexp2(input, other, *, out=None):
    _2_x, _2_y = torch.exp2(input), torch.exp2(other)
    return torch.log2(_2_x+_2_y, out=out), (_2_x, _2_y)


def logical_and():
    ...

def logical_not():
    ...

def logical_or():
    ...

def logical_xor():
    ...


def logit_bwd(dl_dout, input, eps, *, out=None):
    if eps is None:
        dl_dinput = torch.mul(dl_dout, -torch.reciprocal(input*(input-1)), out=out)
    else:
        z, acts = clamp(input, eps, 1-eps)
        dl_dz = torch.mul(dl_dout, torch.reciprocal(z*(z-1)))
        dl_dinput, _, _ = clamp_bwd(dl_dz, *acts, out_mask=[True, False, False], out=[out, None, None])
    return dl_dinput

@register_bwd_op(logit_bwd)
def logit(input, eps=None, *, out=None):
    return torch.logit(input, eps, out=out), (input, eps)


def hypot_bwd(dl_dout, input, other, c, *, out_mask=[True, True], out=[None,None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, input/c, out=out[0])
    else:
        dl_dinput = None
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, other/c, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(hypot_bwd)
def hypot(input, other, *, out=None):
    c = torch.hypot(input, other, out=out)
    return c, (input, other, c)

def i0():
    ...

def igamma():
    ...

def igammac():
    ...


def mul_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None,None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, other, out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[0]:
        dl_dother = torch.mul(dl_dout, input, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(mul_bwd)
def mul(input, other, *, out=None):
    return torch.mul(input, other, out=out), (input, other)

@register_bwd_op
def multiply(input, other, *, out=None):
    return torch.mul(input, other, out=out), (input, other)


def mvlgamma():
    ...

def nan_to_num():
    ...


def neg_bwd(dl_dout, _, *, out=None):
    return torch.neg(dl_dout, out=out)

@register_bwd_op(neg_bwd)
def neg(input, *, out=None):
    return torch.neg(input, out=out), (None,)

@register_bwd_op(neg_bwd)
def negative(input, *, out=None):
    return torch.neg(input, out=out), (None,)


def polygamma():
    ...


def positive_bwd(dl_dout, _,):
    return torch.positive(dl_dout)

def positive(input):
    return torch.positive(input), (None,)


def pow_bwd(dl_dout, input, exponent, y, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, exponent* torch.float_power(input, exponent-1), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dexponent = torch.mul(dl_dout, torch.log(input)*y, out=out[1])
    else:
        dl_dexponent = None
    return dl_dinput, dl_dexponent

@register_bwd_op(pow_bwd)
def pow(input, exponent, *, out=None):
    y = torch.float_power(input, exponent, out=out)
    return y, (input, exponent, y)
    

def rad2deg_bwd(input, _, *, out=None):
    return torch.mul(input, 180/torch.pi, out=out)

@register_bwd_op(rad2deg_bwd)
def rad2deg(input, *, out=None):
    return torch.rad2deg(input, out=out), (None,)


def reciprocal_bwd(dl_dout, y, *, out=None):
    return torch.div(dl_dout, -y*y, out=out)

@register_bwd_op(reciprocal_bwd)
def reciprocal(input, *, out=None):
    y = torch.reciprocal(input, out=out)
    return y, (y,)


def remainder_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None,None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].clone_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, -input.div(other, rounding_mode="trunc"), out=out)
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(reciprocal_bwd)
def remainder(input, other, *, out=None):
    return torch.remainder(input, other, out=out), (input, other,)


def round_bwd(dl_dout, _, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(round_bwd)
def round(input, *, out=None):
    return torch.round(input, out=out)


def rsqrt(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, -0.5/y**3, out=out)

@register_bwd_op(rsqrt)
def rsqrt(input, *, out=None):
    y = torch.rsqrt(input, out=out)
    return y, (y,)


def sigmoid_bwd(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, (1-y), out=out)

@register_bwd_op(sigmoid_bwd)
def sigmoid(input, *, out=None):
    y = torch.sigmoid(input, out=out)
    return y, (y,)


def sign_bwd(dl_dout, _, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(sign_bwd)
def sign(input, *, out=None):
    return torch.sign(input, out=out), (None,)


def sgn_bwd(dl_dout, _, *, out=None):
    #TODO add support for complex input
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(sgn_bwd)
def sgn(input, *, out=None):
    return torch.sgn(input, out=out), (None,)


def sin_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.cos(input), out=out)

@register_bwd_op(sin_bwd)
def sin(input, *, out=None):
    return torch.sin(input, out=out), (input,)


def sinc_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, (torch.cos(torch.pi*input)-torch.sin(torch.pi*input))/(torch.pi*input**2), out=out)

@register_bwd_op(sinc_bwd)
def sinc(input, *, out=None):
    return torch.sinc(input, out=out), (input,)


def sinh_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, torch.cosh(input), out=out)

@register_bwd_op(sinh_bwd)
def sinh(input, *, out=None):
    return torch.sinh(input, out=out), (input,)


def softmax_bwd(dl_dout, y, dim, dtype, *, out=None):
    m = y*dl_dout
    dl_dinput = (m - y*torch.sum(m, dim=dim, keepdim=True)).to(dtype)
    if out is not None:
        dl_dinput = out.copy_(dl_dinput)
    else:
        dl_dinput = dl_dinput.to(dtype)
    return dl_dinput

@register_bwd_op(softmax_bwd)
def softmax(input, dim, dtype):
    y = torch.softmax(input, dim, dtype)
    return y, (y, dim, dtype)


def sqrt_bwd(dl_dout, y, *, out=None):
    return torch.div(dl_dout, 2*y, out=out)

@register_bwd_op(sqrt_bwd)
def sqrt(input, *, out=None):
    y = torch.sqrt(input, out=out)
    return y, (y,)


def square_bwd(dl_dout, input, *, out=None):
    return torch.mul(dl_dout, 2*input, out=out)

@register_bwd_op(square_bwd)
def square(input, *, out=None):
    return torch.square(input, out=out), (input,)


def sub_bwd(dl_dout, input, other, alpha, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        if out[0] is not None:
            dl_dinput = out[0].clone_(dl_dout)
        else:
            dl_dinput = dl_dout
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, -alpha, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(sub_bwd)
def sub(input, other, *, alpha=1, out=None):
    return torch.sub(input, other, alpha=alpha, out=out), (input, other, alpha)

@register_bwd_op(sub_bwd)
def substract(input, other, *, alpha=1, out=None):
    return torch.sub(input, other, alpha=alpha, out=out), (input, other, alpha)


def tan_bwd(dl_dout, input, *, out=None):
    return torch.div(dl_dout, torch.cos(input)**2, out=out)

@register_bwd_op(tan_bwd)
def tan(input, *, out=None):
    return torch.tan(input, out=out), (input,)


def tanh_bwd(dl_dout, y, *, out=None):
    return torch.mul(dl_dout, 1-y*y, out=out)

@register_bwd_op(tanh_bwd)
def tanh(input, *, out=None):
    y = torch.tanh(input, out=out), (input,)
    return y, (y,)


def true_divide():
    ...


def trunc_bwd(dl_dout, _, *, out=None):
    if out is not None:
        dl_dinput = out.zero_()
    else:
        dl_dinput = torch.zeros_like(dl_dout)
    return dl_dinput

@register_bwd_op(trunc_bwd)
def trunc(input, *, out=None):
    return torch.trunc(input, out=out), (None, )

@register_bwd_op(trunc_bwd)
def fix(input, *, out=None):
    return torch.trunc(input, out=out), (None, )


def xlogy_bwd(dl_dout, input, other, *, out_mask=[True, True], out=[None, None]):
    if out_mask[0]:
        dl_dinput = torch.mul(dl_dout, torch.log(other), out=out[0])
    else:
        dl_dinput = None
    
    if out_mask[1]:
        dl_dother = torch.mul(dl_dout, input/other, out=out[1])
    else:
        dl_dother = None
    return dl_dinput, dl_dother

@register_bwd_op(xlogy_bwd)
def xlogy(input, other, *, out=None):
    return torch.xlogy(input, other, out=out), (input, other)