"""
Variational Network

Reference:
* Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. Learning a variational network for reconstruction of accelerated MRI data. Magn Reson Med 2018;79:3055-3071.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from utils import r2c, c2r
from models import mri, unet

# %%
class data_consistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self,
                curr_x: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        A = mri.SenseOp(coil, mask)
        grad = A.adj(A.fwd(curr_x)) - x0

        next_x = curr_x - self.lam * grad

        return next_x

# %%
class VarNet(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        self.dc = data_consistency()
        self.dw = unet.Unet(2, 2, num_pool_layers=n_layers)

    def forward(self, x0, coil, mask):

        x0 = r2c(x0, axis=1)
        xk = x0.clone()

        for c in range(self.n_cascades):

            x = self.dc(xk, x0, coil, mask)
            x = x - r2c(self.dw(c2r(x, axis=1)), axis=1)

        return c2r(x, axis=1)
