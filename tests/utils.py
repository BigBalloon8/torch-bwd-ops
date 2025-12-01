import torch
import torch_ops

def test_grad_1in1out(fwd_op, fwd_inputs):
    for input in fwd_inputs:
        if isinstance(input, torch.Tensor):
            input.requires_grad = True
    out, activations = fwd_op(*fwd_inputs)
    if out.dim() == 0:
        out.backward()
    else:
        out.sum().backward()
    true_grad = [input.grad for input in fwd_inputs if isinstance(input, torch.Tensor)][0]
    calc_grad = fwd_op.bwd(torch.ones_like(out), *activations)
    torch.testing.assert_close(calc_grad, true_grad)

def get_all_functions():
    funcs = []
    for f in dir(torch_ops):
        if callable(getattr(torch_ops, f)):
            funcs.append(f)

class Dim:
    pass

class Dim1:
    pass

class Kth:
    pass

class Indices:
    pass