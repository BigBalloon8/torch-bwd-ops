import torch

from torch_ops._wrappers import register_bwd_op


addbmm = [("n","p"), ("b","n","m"), ("b","m","p"), {"beta":[], "alpha":[]}]

addmm = [("n","p"), ("n","m"), ("m","p"), {"beta":[], "alpha":[]}]

addmv = [("n",), ("n","m"), ("m",), {"beta":[], "alpha":[]}]

addr = [("n", "m"), ("n",), ("m",), {"beta":[], "alpha":[]}]

baddbmm = [("b","n","p"), ("b","n","m"), ("b","m","p"), {"beta":[], "alpha":[]}]

bmm = [("b","n","m"), ("b","m","p")]

chain_matmul = [("n", "m"), ("m", "p"), ("p", "s"), ("s", "t"), ("t", "v")]

cholesky = ...

cholesky_inverse = ...

cholesky_solve = ...

dot = [("n",), ("n",)]

geqrf = ...

inner = [(...,"n"), (...,"n")]

inverse = [("n","n")]

det = [("n", "n")]

logdet = [("n", "n")]


slogdet = ...

lu = ...

lu_solve = ...

lu_unpack = ...

mm = [("n","m"), ("m","p")]


def matmul(input, other, *, out=None):
    return torch.matmul(input, other, out=out), (input, other)

matrix_power = [("n","n"), []]

matrix_exp = ...

mv = [("n","m"), ("m",)]

orgqr = ...

ormqr = ...

outer = [("n",), ("m",)]

ger = outer

pinverse = ...

qr = ...

svd = ...

svd_lowrank = ...

pca_lowrank = ...

lobpcg = ...

trapz = ...

trapezoid= ...

cumulative_trapezoid = ...

triangular_solve = ...

vdot = [("n",),("n",)]

