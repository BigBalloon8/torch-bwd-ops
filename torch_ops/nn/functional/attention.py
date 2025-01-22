import torch
import math

from torch_ops._wrappers import register_bwd_op
from torch_ops.nn.functional.dropout import dropout
from torch_ops.nn.functional.activation import softmax

def scaled_dot_product_attention_bwd(dl_dout, query, key, value, scale_factor, enable_gqa, d_mask, softmax_att_weight, *, out_mask=[True, True, True], out=[None, None, None]):
    if not enable_gqa:
        if out_mask[2]:
            final_softmax_att_weight = d_mask*softmax_att_weight
            dl_dvalue = torch.matmul(softmax_att_weight.transpose(-1,-2), dl_dout, out=out[2])
        else:
            dl_dvalue = None
        
        if out_mask[0] or out_mask[1]:
            dl_dfinal_softmax_att_weight = dl_dout @ value.transpose(-1,-2)
            dl_dsoftmax_att_weight = dropout.bwd(dl_dfinal_softmax_att_weight, d_mask)
            dl_datt_weight = softmax.bwd(dl_dsoftmax_att_weight, softmax_att_weight, dim=-1, dtype=None) * scale_factor

            if out_mask[0]:
                dl_dquery = torch.matmul(dl_datt_weight, key, out=out[0])
            else:
                dl_dquery = None

            if out_mask[1]:
                dl_dkey = torch.matmul(query.transpose(-1,-2), dl_datt_weight, out=out[1]).transpose(-1,-2)
            else:
                dl_dkey = None
        else:
            dl_dquery = None
            dl_dkey = None
    else:
        d_shape = dl_dout.shape
        new_Q_shape = d_shape[:-3] + (d_shape[-3]//key.size(-3), key.size(-3), d_shape[-2], d_shape[-1])
        dl_dout = dl_dout.reshape(new_Q_shape)

        if out_mask[2]:
            final_softmax_att_weight = d_mask*softmax_att_weight
            dl_dvalue = torch.vmap(torch.vmap(lambda sqkt, dldatt: torch.bmm(sqkt.transpose(-1,-2), dldatt), in_dims=-3, out_dims=-3), in_dims=-3, out_dims=-3)(final_softmax_att_weight, dl_dout)
            dl_dvalue = torch.sum(dl_dvalue, -4, out=out[2])
        else:
            dl_dvalue = None
        
        if out_mask[0] or out_mask[1]:
            #print(dl_dout.shape, value.shape)
            dl_dfinal_softmax_att_weight = torch.vmap(lambda dldatt, v: torch.vmap(lambda _dldatt: torch.bmm(_dldatt, v.transpose(-1,-2)), in_dims=-3, out_dims=-3)(dldatt), in_dims=-3, out_dims=-3)(dl_dout, value)
            #print(dl_dfinal_softmax_att_weight.shape, d_mask.shape)
            dl_dsoftmax_att_weight = dropout.bwd(dl_dfinal_softmax_att_weight, d_mask)
            #print(dl_dsoftmax_att_weight.shape, softmax_att_weight.shape)
            dl_datt_weight = softmax.bwd(dl_dsoftmax_att_weight, softmax_att_weight, dim=-1, dtype=None) * scale_factor

            if out_mask[0]:
                dl_dquery = torch.vmap(lambda dldqkt, k: torch.vmap(lambda _dldqkt: torch.bmm(_dldqkt, k), in_dims=-3, out_dims=-3)(dldqkt), in_dims=-3, out_dims=-3)(dl_datt_weight, key)
                dl_dquery = dl_dquery.flatten(-4,-3)
                if out[0] is not None:
                    dl_dquery = out[0].copy_(dl_dquery)
            else:
                dl_dquery = None
    
            if out_mask[1]:
                dl_dkeyt = torch.vmap(torch.vmap(lambda q, dldqkt: torch.bmm(q.transpose(-1,-2), dldqkt), in_dims=-3, out_dims=-3), in_dims=-3, out_dims=-3)(query, dl_datt_weight)
                dl_dkeyt = torch.sum(dl_dkeyt, -4, out=out[1])
                dl_dkey = dl_dkeyt.transpose(-1,-2)
            else:
                dl_dkey = None
        else:
            dl_dquery = None
            dl_dkey = None

    return dl_dquery, dl_dkey, dl_dvalue

@register_bwd_op(scaled_dot_product_attention_bwd)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask

    if not enable_gqa:
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        softmax_att_weight = torch.softmax(attn_weight, dim=-1)
        final_softmax_att_weight, d_mask = dropout(softmax_att_weight, dropout_p, training=True)
        return final_softmax_att_weight @ value, (query, key, value, scale_factor, enable_gqa, *d_mask, softmax_att_weight)
    else:
        Q_shape = query.shape
        new_Q_shape = Q_shape[:-3] + (Q_shape[-3]//key.size(-3), key.size(-3), Q_shape[-2], Q_shape[-1])
        query = query.reshape(new_Q_shape)
        attn_weight = torch.vmap(lambda q, k: torch.vmap(lambda _q: torch.bmm(_q, k.transpose(-1,-2)), in_dims=-3, out_dims=-3)(q), in_dims=(-3,-3), out_dims=-3)(query, key) * scale_factor
        attn_weight += attn_bias
        softmax_att_weight = torch.softmax(attn_weight, dim=-1)
        final_softmax_att_weight, d_mask = dropout(softmax_att_weight, dropout_p, training=True)
        out = torch.vmap(lambda sqkt, v: torch.vmap(lambda _sqkt: torch.bmm(_sqkt, v), in_dims=-3, out_dims=-3)(sqkt), in_dims=-3, out_dims=-3)(final_softmax_att_weight, value)
        out = out.flatten(-4,-3)
        return out, (query, key, value, scale_factor, enable_gqa, *d_mask, softmax_att_weight)
