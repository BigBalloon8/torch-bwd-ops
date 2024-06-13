import torch
import torch_ops
import time



x = torch.randn(10000,10000)
dout = torch.randn(10000,10000)
dx = torch_ops.abs.bwd(dout, x)

s = time.time()
out, acts = torch_ops.abs(x)
dx = torch_ops.abs.bwd(dout, *acts)
print(time.time()- s)