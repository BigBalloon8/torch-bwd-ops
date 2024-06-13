import torch
import torch_ops
import time



a = torch.ones(100000,10000)+1
b = (torch.arange(10) + 1).repeat(10_000_000).reshape(10000,10000)

s = time.time()
(a**0.5)**3
i = time.time()
torch.pow(a, 1.5)
f = time.time()
print(i-s, f-i)