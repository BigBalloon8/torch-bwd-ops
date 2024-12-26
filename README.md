# Torch Ops 

This library has been designed to provide researchers with a way to functionalally call backward pass operations.

```python
import torch
import torch_ops

x = torch.randn(64,64)
out = torch.empty_like(x)

# f(x) = clip(cos(x), -0.9, 0.9)
h, activations1 = torch_ops.cos(x)
y, activations2 = torch_ops.clip(h, min=-0.9, max=0.9, out=out)

dl_dy = torch.ones_like(y)
dl_dx_buffer = torch.empty_like(x)

#f'(x)
dl_dh = torch_ops.clip.bwd(dl_dy, *activations2)
dl_dx = torch_ops.cos.bwd(dl_dh, *activations1, out=dl_dx_buffer)
```

