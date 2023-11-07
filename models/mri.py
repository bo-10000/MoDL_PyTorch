"""
This module implements MRI operators

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np

import torch
import torch.nn as nn
from typing import Optional, Tuple


def fftc(input: torch.Tensor | np.ndarray,
         axes: Optional[Tuple] = (-2, -1),
         norm: Optional[str] = 'ortho'):

        if isinstance(input, np.ndarray):
            tmp = np.fft.ifftshift(input, axes=axes)
            tmp = np.fft.fftn(tmp, axes=axes, norm=norm)
            output = np.fft.fftshift(tmp, axes=axes)

        elif isinstance(input, torch.Tensor):
            tmp = torch.fft.ifftshift(input, dim=axes)
            tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
            output = torch.fft.fftshift(tmp, dim=axes)

        return output

def ifftc(input: torch.Tensor | np.ndarray,
          axes: Optional[Tuple] = (-2, -1),
          norm: Optional[str] = 'ortho'):

        if isinstance(input, np.ndarray):
            tmp = np.fft.ifftshift(input, axes=axes)
            tmp = np.fft.ifftn(tmp, axes=axes, norm=norm)
            output = np.fft.fftshift(tmp, axes=axes)

        elif isinstance(input, torch.Tensor):
            tmp = torch.fft.ifftshift(input, dim=axes)
            tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
            output = torch.fft.fftshift(tmp, dim=axes)

        return output


class SenseOp():
    """
    Sensitivity Encoding (SENSE) Operators

    Reference:
        * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
          Advances in sensitivity encoding with arbitrary k-space trajectories.
          Magn Reson Med (2001).
    """
    def __init__(self,
                 coil: torch.Tensor | np.ndarray,
                 mask: torch.Tensor | np.ndarray,
                 traj: Optional[torch.Tensor | np.ndarray] = None):
        """
        Args:
            coil: [N_batch, N_coil, N_y, N_x]
            mask: [N_batch, N_y, N_x]
            traj:
        """

        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if (traj is not None) and isinstance(traj, np.ndarray):
            traj = torch.from_numpy(traj)


        self.coil = coil
        self.mask = mask
        self.traj = traj

    def fwd(self, input) -> torch.Tensor:
        """
        SENSS Forward Operator: from image to k-space
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        N_batch, N_coil, N_y, N_x = self.coil.shape

        coils = torch.swapaxes(self.coil, 0, 1)
        coils = coils * input
        kfull = fftc(coils, norm='ortho')

        if self.traj is None:
            # Cartesian sampling
            output = torch.swapaxes(self.mask * kfull, 0, 1)
        else:
            # TODO: Radial sampling
            None

        return output

    def adj(self, input) -> torch.Tensor:
        """
        SENSE Adjoint Operator: from k-space to image
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        kfull = torch.swapaxes(input, 0, 1)

        if self.traj is None:
            # Cartesian sampling
            kmask = torch.swapaxes(self.mask * kfull, 0, 1)
            imask = ifftc(kmask, norm='ortho')
        else:
            # TODO: Radial sampling
            None

        output = torch.sum(imask * self.coil.conj(), dim=1)

        return output

class SenseSp():
    """
    Implementation of the SENSE Operator based on SigPy.
    """