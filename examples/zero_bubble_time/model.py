from typing import Sequence, Dict, Optional
from contextlib import contextmanager
import time
from functools import wraps
from abc import ABC, abstractmethod
import math

import torch
import torch.distributed as dist
from torch.nn import init

import torch_ops

from loss import Loss

@contextmanager
def nvtx_profile(name):
    yield
    return
    torch.cuda.synchronize()
    #print(name)
    torch.cuda.nvtx.range_push(name)
    yield
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

class cleanup_act(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, func):
        @wraps(func)
        def wrapper(_self, inter):
            out = func(_self, inter)
            if _self.multi_stage and not inter:
                #print(_self.__class__.__name__, self.args)
                for arg in _self.acts:
                    #print(f"Setting {_self.__class__.__name__}.{arg} to []")
                    setattr(_self, arg, [])
            return out
        return wrapper

def multi_stage_wrapper(func):
    # designed to wrap backward_p1
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.multi_stage: # or dist.get_rank() == 0:
            out = func(self, *args, **kwargs)
            self.backward_p2(inter=False)
            return out
        else:
            return func(self, *args, **kwargs)
    return wrapper

class expose_params(object):
    def __init__(self, p_and_g: dict={}, acts:list=[]):
        self.p_and_g = p_and_g 
        self.acts = acts
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            _self = args[0]
            _self.params = {}
            _self.grads = {}
            _self.acts = []
            for p, g in self.p_and_g.items():
                if not hasattr(_self, p) or getattr(_self, p) is None:
                    continue
                _self.params[p] = getattr(_self, p)
                _self.grads[p] = getattr(_self, g)
            _self.acts = self.acts
            return out

        return wrapper

class Layer(ABC):
    def __init__(self):
        self.multi_stage = True
        self.training = True
        self.params: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.acts = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass
    
    def init_params(self):
        pass

    @abstractmethod
    def backward_p1(self):
        pass
    
    def backward_p2(self, inter=False):
        #p2 is for calculating param grads if theres no params can just pass
        pass

    def update(self):
        pass
    
    def clear_acts(self):
        if self.acts:
            for act in self.acts:
                setattr(self, act, [])
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).clear_acts()
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.clear_acts()
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                         sub_layer.clear_acts()

    def to_(self, arg):
        if hasattr(self, "params"):
            for k in self.params.keys():
                self.params[k] = self.params[k].to(arg)
            for k in self.grads.keys():
                self.grads[k] = self.grads[k].to(arg)
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).to_(arg)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.to_(arg)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                         sub_layer.to_(arg)
    
    def multi_stage_set(self, _bool):
        self.multi_stage = _bool
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).multi_stage_set(_bool)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.multi_stage_set(_bool)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.multi_stage_set(_bool)
    
    def zero_grad(self):
        if self.grads:
            for k in self.grads.keys():
                self.grads[k].zero_()
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).zero_grad()
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.zero_grad()
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.zero_grad()
    
    def _get_model_sub_layers(self, sublayers_list: list):
        if self.grads:
            sublayers_list.append(self)
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item)._get_model_sub_layers(sublayers_list)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer._get_model_sub_layers(sublayers_list)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer._get_model_sub_layers(sublayers_list)
    
    def get_num_params(self, num_elm_list:list):
        if self.params:
            for param in self.params.values():
                num_elm_list.append(torch.numel(param))
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).get_num_params(num_elm_list)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.get_num_params(num_elm_list)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.get_num_params(num_elm_list)

def dense_params(in_features, out_features, bias = True, factory_kwargs={}):
    w = torch.empty(in_features, out_features, **factory_kwargs)
    init.kaiming_uniform_(w, a=math.sqrt(5))
    if bias:
        b = torch.empty(out_features, **factory_kwargs)
        fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(b, -bound, bound)
    else:
        b = None
    return w, b

def embedding_params(num_embeddings, embedding_dim, factory_kwargs={}):
    w = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
    init.normal_(w)
    return w

class Activation(Layer):
    """
    Used for Activation functions that dont have parameters
    """

    def backward_p2(self, inter=False):
        pass

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, device:str = "cuda", dtype=torch.float32):
        """
        Initialize a Dense Layer

        Args:
            input_size (int): size of input vector
            output_size (int): size of output vector
            bias (bool): whether or not to include bias
            device (str): device to use for computation (cpu or cuda)
        
        Attributes:
            weights (torch.Tensor): weights of the dense layer
            bias (torch.Tensor): bias of the dense layer
            weights_g (torch.Tensor): gradients of the weights of the dense layer
            bias_g (torch.Tensor): gradients of the bias of the dense layer
            inputs (torch.Tensor): inputs of the dense layer
            dL_dout (torch.Tensor): derivative of the loss with respect to the dense layers output
        
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias_ = bias
        self.device = device
        self.dtype = dtype
    
    @expose_params({"weights": "weights_g", "bias": "bias_g"}, ["inputs", "dL_dout"])
    def init_params(self):
        self.weights, self.bias = dense_params(self.input_size, self.output_size, self.bias_, {"device":self.device, "dtype":self.dtype})
        
        self.weights_g = torch.zeros_like(self.weights, device=self.device)
        self.bias_g = torch.zeros_like(self.bias, device=self.device) if self.bias_ else None

        self.activations = []
        self.dL_dout = []
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dense layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying dense layer
        """
        out, acts = torch_ops.nn.functional.linear(x, self.weights, self.bias)
        self.acts.append(acts)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the dense layers output and returns the derivative of the loss with respect to the dense layers input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dense layers output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dense layers input
        """
        self.dL_dout.append(dL_dout)
        acts = self.activations.pop(0)
        dl_din, _, _ =  torch_ops.nn.functional.linear_bwd(dL_dout, *acts, out_mask=[True, False, False])
        self.activations.append(acts)
        return dl_din
    
    @cleanup_act("dL_dout", "acts")
    def backward_p2(self, inter=False):
        """
        Calculates the gradients of the dense layer's parameters
        """
        
        #n = len(self.inputs) if not inter else 1
        if self.multi_stage and not inter:
            dL_dout = torch.cat(self.dL_dout, dim=0)
            inputs = torch.cat([a[0] for a in range(len(self.acts))])
        else:
            dL_dout = self.dL_dout.pop(0)
            inputs = self.acts.pop(0)[0]

        _, dl_dweight, dl_dbias = torch_ops.nn.functional.linear_bwd(dL_dout, inputs, self.weights)
        
        if inputs.ndim == 2:
            self.weights_g[:] += torch.sum(dl_dweight, dim=0)
        elif inputs.ndim == 3:
            self.weights_g[:] += torch.sum(dl_dweight, dim=tuple(range(dL_dout.ndim)[:-2]))
        
        if self.bias is not None:
            self.bias_g[:] += torch.sum(dl_dbias, dim=tuple(range(dL_dout.ndim)[:-1]))
            
            #self.dL_dout = None

class Dropout(Layer):
    @expose_params(acts=["p_mask"])
    def __init__(self, p: Optional[float]=0.1):
        """
        Initialize a dropout layer

        Args:
            p (Optional[float]): Probability of neurons being dropped. Defaults to 0.1.
        
        Attributes:
            p (Optional[float]): Probability of neurons being dropped. Defaults to 0.1.
            p_mask (torch.Tensor): Mask for the dropout layer
            training (bool): Turned to False during inference in order to turn off dropout
        """
        super().__init__()
        self.p = p
        self.training = True
        self.acts = []
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dropout layer. 
        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output tensor after applying the dropout
        """
        #torch.manual_seed(0)
        out, acts = torch_ops.nn.functional.dropout(x, self.p, self.training)
        self.acts.append(acts)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the dropouts output and returns the derivative of the loss with respect to the dropouts input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dropout's output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dropout's input
        """
        if self.p==0:
            return dL_dout
        if self.p==1:
            return torch.zeros_like(dL_dout)
        return torch_ops.nn.functional.dropout_bwd(dL_dout, *self.acts.pop(0))


class Embeddings(Layer):
    def __init__(self, num_embeddings, dim, device, dtype=torch.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.device = device
        self.dtype = dtype
    
    @expose_params({"weights": "weights_g"}, ["inputs", "dL_dout"])
    def init_params(self):
        self.weights = embedding_params(self.num_embeddings, self.dim, {"device": self.device, "dtype":self.dtype})
        self.weights_g = torch.zeros_like(self.weights, device=self.device)
        self.acts = []
        self.dL_dout = []
    
    def forward(self, x):
        out, acts = torch_ops.nn.functional.embedding(x, self.weights)
        self.acts.append(acts)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        # Embedding is the first operation so doesnt need a backward pass
        self.dL_dout.append(dL_dout)
        pass
    
    @cleanup_act("dL_dout", "inputs")
    def backward_p2(self, inter=False):
        n = len(self.dL_dout) if not inter else 1
        for _ in range(n):
            dL_dout = self.dL_dout.pop(0)
            act = self.acts.pop(0)
            self.weights_g += torch_ops.nn.functional.embedding_bwd(dL_dout, *act)


class SiLU(Activation):
    @expose_params(acts=["inputs"])
    def __init__(self):
        super().__init__()
        self.acts = []

    def forward(self, x: torch.Tensor):
        out, acts = torch_ops.nn.functional.silu(x)
        self.acts.append(acts)
        return out

    def backward_p1(self, dL_dout: torch.Tensor):
        acts = self.acts.pop(0)
        return torch_ops.nn.functional.silu_bwd(dL_dout, *acts)


class llamaFF(Layer):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier = None, device:str = "cuda", dtype=torch.float32):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.device = device
        self.dtype = dtype
    
    @expose_params(acts=["l2", "silu_out"])
    def init_params(self):
        self.linears = [
            Dense(self.dim, self.hidden_dim, bias=False, device=self.device, dtype=self.dtype),
            Dense(self.hidden_dim, self.dim, bias=False, device=self.device, dtype=self.dtype),
            Dense(self.dim, self.hidden_dim, bias=False, device=self.device, dtype=self.dtype)
        ]
        for l in self.linears:
            l.init_params()
        
        self.silu = SiLU()

        self.l2 = []
        self.silu_out = []

    def forward(self, x: torch.Tensor):
        l0 = self.linears[0](x)
        l2 = self.linears[2](x)
        self.l2.append(l2)
        silu_out = self.silu(l0)
        self.silu_out.append(silu_out)
        return self.linears[1](silu_out*l2)
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout:torch.Tensor):
        dL_dl2in = self.linears[1].backward_p1(dL_dout)
        dL_dl0 = self.silu.backward_p1(dL_dl2in * self.l2.pop(0))
        dL_dx1 = self.linears[0].backward_p1(dL_dl0)
        dL_dx2 = self.linears[2].backward_p1(dL_dl2in * self.silu_out.pop(0)) 
        return dL_dx1 + dL_dx2

class TransformerPPBlock(Layer):
    def __init__(self, dim, num_heads, num_kv_heads, max_seqlen, theta=10000.0, p=0.1, device="cuda", dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seqlen = max_seqlen
        self.theta = theta
        self.p = p

        self.device = device
        self.dtype = dtype
    
    def init_params(self):
        self.attention = GroupedMultiQueryAttention(self.dim, self.num_heads, self.num_kv_heads, self.max_seqlen, theta=self.theta, p=self.p, device=self.device, dtype=self.dtype)
        self.ff = llamaFF(self.dim, self.dim*4, 256, device=self.device, dtype=self.dtype)
        self.att_norm = NLPRMSNorm(-1, self.dim, device=self.device, dtype=self.dtype)
        self.ff_norm = NLPRMSNorm(-1, self.dim, device=self.device, dtype=self.dtype)
        self.att_dropout = Dropout(self.p)
        self.ff_dropout = Dropout(self.p)

        mask = torch.zeros(self.max_seqlen, self.max_seqlen, device=self.device, dtype=self.dtype)
        self.mask = mask.masked_fill(torch.triu(torch.ones(self.max_seqlen, self.max_seqlen, device=self.device, dtype=self.dtype), diagonal=1).bool(), float('-inf'))
    
        self.attention.init_params()
        self.ff.init_params()
        self.att_norm.init_params()
        self.ff_norm.init_params()

    def forward(self, x):
        n_x = self.att_norm(x)
        if self.training:
            x = self.attention(n_x, self.mask) + x
        else:
            x = self.attention(n_x) + x
        h = self.att_dropout(x)
        x = self.ff(self.ff_norm(h)) + h
        return self.ff_dropout(x)

    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        #with nvtx_profile("ff_dropout"):
        dL_dff = self.ff_dropout.backward_p1(dL_dout)
        #with nvtx_profile("ff_back"):
        dL_h = self.ff_norm.backward_p1(self.ff.backward_p1(dL_dff)) + dL_dff
        #with nvtx_profile("att_dropout"):
        dL_datt = self.att_dropout.backward_p1(dL_h)
        dL_dnorm = self.attention.backward_p1(dL_datt)
        #with nvtx_profile("att_norm"):
        out = self.att_norm.backward_p1(dL_dnorm) + dL_datt
        return out


class Model:
    def __init__(self):
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.rank_rep_0 = 0  # for chimera
        self.rank_rep_1 = 0  # for chimera
        self.layers: list[Layer] = []
        self._2bp = True
        self.criterion: Loss = ...
        self.pipe_algo = "none"
        self.x = ...
        self.dL_dout = ...
        self.f_recv_buffer = ...
        self.b_recv_buffer = ...
        self.sub_layers: list[Layer] = []
        self.streams: list[torch.cuda.Stream] = []
        self.t0 = []
        self.t1 = []
        self.t2 = []
        self.bwd_p2_times = []
    
    def _rank_0_fwd(self, x:torch.Tensor):
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            for layer in self.layers:
                #with nvtx_profile(layer.__class__.__name__):
                x = layer(x)
            dist.isend(x.contiguous(), self.rank+1)
    
    def _rank_n_fwd(self):
        dist.recv(self.f_recv_buffer, self.rank-1)
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            for layer in self.layers:
                #with nvtx_profile(layer.__class__.__name__):
                x = layer(x)
            dist.isend(x.contiguous(), self.rank+1)
    
    def _rank_N_fwd(self, y):
        dist.recv(self.f_recv_buffer, self.rank-1)
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            x = self.f_recv_buffer.clone()
            for layer in self.layers:
                #with nvtx_profile(layer.__class__.__name__):
                x = layer(x)
            return self.criterion(x, y)
    
    def _rank_0_bwd_p1(self):
        dist.recv(self.b_recv_buffer, self.rank+1)
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                #with nvtx_profile(layer.__class__.__name__):
                dL_dout = layer.backward_p1(dL_dout)
    
    def _rank_n_bwd_p1(self):
        dist.recv(self.b_recv_buffer, self.rank+1)
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                #with nvtx_profile(layer.__class__.__name__):
                dL_dout = layer.backward_p1(dL_dout)
            dist.isend(dL_dout.contiguous(), self.rank-1)
    
    def _rank_N_bwd_p1(self):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.criterion.backward()
            for layer in self.layers[::-1]:
                #with nvtx_profile(layer.__class__.__name__):
                dL_dout = layer.backward_p1(dL_dout)
            if not self._2bp and isinstance(self, Transformer): # needed for memory
                torch.cuda.empty_cache()
            dist.isend(dL_dout.contiguous(), self.rank-1)
    
    def _backward_p2(self, inter=False):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd_p2"):
            if self.device == "cuda":
                for i, layer in enumerate(self.sub_layers):
                    #with torch.cuda.stream(self.streams[i]):
                    #if layer.__class__.__name__ == "Conv2D":
                    #   extra_info = f" {layer.}"
                    #with nvtx_profile(layer.__class__.__name__):
                    layer.backward_p2(inter=inter)
            else:
                for layer in self.layers:
                    layer.backward_p2()
            torch.cuda.synchronize()
                
    def chunk_data(self, x, n):
        if isinstance(x, Sequence):
            map_gen = map(lambda _x: torch.chunk(_x, n), x)
            return type(x)(zip(*map_gen))
        else:
            return torch.chunk(x, n)

    def train_step(self, x, y):
        if dist.get_world_size() == 1:
            for layer in self.layers:
                x = layer(x)
            loss = self.criterion(x, y)
            dl_dlout = self.criterion.backward()
            for layer in self.layers[::-1]:
                dl_dlout = layer.backward_p1(dl_dlout)
            if self._2bp:
                self._backward_p2()
            return [loss]

        losses = []
        if self.pipe_algo == "none":
            if self.rank == 0:
                #self.t0.append(time.time_ns())
                self._rank_0_fwd(x)
                self._rank_0_bwd_p1()
                #torch.cuda.synchronize()
                #self.t2.append(time.time_ns())
            elif self.rank == self.size - 1:
                loss = self._rank_N_fwd(y)
                #torch.cuda.synchronize()
                #self.t1.append(time.time_ns())
                self._rank_N_bwd_p1()
                losses.append(loss)
            else:
                self._rank_n_fwd()
                self._rank_n_bwd_p1()
            if self._2bp:
                #torch.cuda.synchronize()
                #bp2_t0 = time.time_ns()
                self._backward_p2()
                #torch.cuda.synchronize()
                #self.bwd_p2_times.append(time.time_ns()-bp2_t0)

        
        elif self.pipe_algo == "gpipe":
            if self.rank == 0:
                micro_batches = self.chunk_data(x, self.size)
            if self.rank == self.size - 1:
                micro_batches = self.chunk_data(y, self.size)
            step = range(self.size)
            
            if self.rank == 0:
                for mb in micro_batches:
                    self._rank_0_fwd(mb)
                for _ in step:
                    self._rank_0_bwd_p1()
                
            elif self.rank == self.size - 1:
                for mb in micro_batches:
                    loss = self._rank_N_fwd(mb)
                for _ in step:
                    self._rank_N_bwd_p1()
                losses.append(loss)

            else:
                for _ in step:
                    self._rank_n_fwd()
                for _ in step:
                    self._rank_n_bwd_p1()
            
            if self._2bp:
                self._backward_p2()
        
        
        elif self.pipe_algo == "1f1b-2":
            if self.rank == 0:
                micro_batches = list(self.chunk_data(x, self.size*2))
            elif self.rank == self.size - 1:
                micro_batches = list(self.chunk_data(y, self.size*2))
            else:
                micro_batches = list(range(self.size*2))
            # Warmup
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_fwd(micro_batches.pop(0))
                elif self.rank == self.size - 1:
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_fwd()
            #Full Overlap
            while len(micro_batches) > 0:
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    self._rank_0_fwd(micro_batches.pop(0))
                    
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_bwd_p1()
                    self._rank_n_fwd()
            # Cool Down
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
                else:
                    self._rank_n_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
        
            if self._2bp:
                self._backward_p2()
        

        elif self.pipe_algo == "1f1b-1":
            if self.rank == 0:
                micro_batches = list(self.chunk_data(x, self.size))
            elif self.rank == self.size - 1:
                micro_batches = list(self.chunk_data(y, self.size))
            else:
                micro_batches = list(range(self.size))
            # Warmup
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_fwd(micro_batches.pop(0))
                elif self.rank == self.size - 1:
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_fwd()
            
            while len(micro_batches) > 0:
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    self._rank_0_fwd(micro_batches.pop(0))
                    
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_bwd_p1()
                    self._rank_n_fwd()
            
            for i in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
                else:
                    self._rank_n_bwd_p1()
                    if self._2bp:
                        self._backward_p2(inter=True)
            
            if self._2bp and self.rank != 0:
                self._backward_p2()
        else:
            raise NotImplementedError(f"Pipe algo {self.pipe_algo} not implemented please select none|gpipe|1f1b-1|1f1b-2")
 
        dist.barrier()
        return losses

    def update(self):
        with nvtx_profile(f"Rank {dist.get_rank()} Update"):
            for layer, s in zip(self.sub_layers, self.streams):
                #with torch.cuda.stream(s):
                layer.update()
    
    def multi_stage(self, _bool):
        for layer in self.layers:
            layer.multi_stage_set(_bool)
        self._2bp = _bool
    
    def zero_grad(self):
        with nvtx_profile(f"Rank {dist.get_rank()} Update"):
            for layer in self.layers:
                layer.zero_grad()
        dist.barrier()
    
    def zero_act(self):
        for layer in self.layers:
            layer.clear_acts()

    def save(self, path):
        pass

    def load(self, path):
        pass

    def to(self, arg):
        raise NotImplementedError("To not supported yet")
        for layer in self.layers:
            layer.to_(arg)
    
    def get_num_params(self):
        tensor_sizes = []
        for layer in self.layers:
            layer.get_num_params(tensor_sizes)
        tensor = torch.tensor(sum(tensor_sizes), device=self.device).clone()
        dist.all_reduce(tensor)
        return tensor.item()

    
class Transformer(Model):
    def __init__(self, dim_size, num_heads, num_kv_heads, max_seqlen, num_blocks, vocab_size, criterion, pipe_algo="none", device="cuda", dtype=torch.float32):
        super().__init__()
        self.block_kwargs = {
            "dim": dim_size,
            "num_heads": num_heads, 
            "num_kv_heads": num_kv_heads, 
            "max_seqlen": max_seqlen, 
            "device": device,
            "dtype": dtype
            }
        self.embedding_kwargs ={
            "num_embeddings": vocab_size,
            "dim": dim_size, 
            "device": device,
            "dtype": dtype
        }
        self.num_blocks = num_blocks
        self.device = device
        self.dtype = dtype
        self.criterion = criterion
        self.pipe_algo = pipe_algo
        
    
    def init_params(self, gbs, input_shape):
        self.layers = []
        num_local_blocks = self.num_blocks // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers.append(layer)
        
        if dist.get_rank() == 0:
            self.layers.insert(0, Embeddings(**self.embedding_kwargs))
            self.layers[0].init_params()
        if dist.get_rank() == dist.get_world_size() - 1:
            self.layers.append(layers.NLPRMSNorm(-1, self.block_kwargs["dim"], device=self.block_kwargs["device"], dtype=self.dtype))
            self.layers.append(Dense(self.block_kwargs["dim"], self.embedding_kwargs["num_embeddings"], bias=False, device=self.device, dtype=self.dtype))
            self.layers[-1].init_params()
            self.layers[-2].init_params()
        
        if self.pipe_algo in ("gpipe", "1f1b-1"):
            mb_size = (gbs // dist.get_world_size(),)
        elif self.pipe_algo == "1f1b-2":
            mb_size = (gbs // (dist.get_world_size() * 2),)
        else:
            mb_size = (gbs,)

        self.f_recv_buffer = torch.zeros(mb_size + input_shape, device=self.device, dtype=self.dtype)
        self.b_recv_buffer = self.f_recv_buffer

        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)   
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        torch.cuda.synchronize()

