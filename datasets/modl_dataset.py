import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from utils import c2r

class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.sigma = sigma

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]
        x0 = undersample(gt, csm, mask, self.sigma)
        # mask = np.stack([mask, mask])
        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(c2r(csm)), torch.from_numpy(mask)
        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Mask'])
        return num_data

def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine