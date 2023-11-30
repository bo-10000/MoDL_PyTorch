import math
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
    def write(self, log_message, verbose=True):
        with open(self.log_dir, 'a') as f:
            f.write(log_message)
            f.write('\n')
        if verbose:
            print(log_message)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#math ================================
def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

def fft_new(image, ndim, normalized=False):
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))

    image = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image


def ifft_new(image, ndim, normalized=False):
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    image = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the third last two channels ((batch), (coil), 2, nx, ny).
    if len(a.size()) == 3:
        return torch.cat(((a[0] * b[0] - a[1] * b[1]).unsqueeze(0),
                          (a[0] * b[1] + a[1] * b[0]).unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat(((a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]).unsqueeze(1),
                          (a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]).unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat(((a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]).unsqueeze(2),
                          (a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]).unsqueeze(2)), dim=2)

def complex_conj(a):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    if len(a.size()) == 3:
        return torch.cat((a[0].unsqueeze(0), -a[1].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0].unsqueeze(1), -a[:, 1].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0].unsqueeze(2), -a[:, :, 1].unsqueeze(2)), dim=2)

#metrics ==================================================
def psnr_batch(y_batch, y_pred_batch):
    #calculate psnr for every batch and return mean
    mean_psnr = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_psnr += psnr(y, y_pred, y.max())
    return mean_psnr / y_batch.shape[0]

def psnr(y, y_pred, MAX_PIXEL_VALUE=1.0):
    rmse_ = rmse(y, y_pred)
    if rmse_ == 0:
        return float('inf')
    return 20 * math.log10(MAX_PIXEL_VALUE/rmse_+1e-10)

def ssim_batch(y_batch, y_pred_batch):
    mean_ssim = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_ssim += ssim(y, y_pred)
    return mean_ssim / y_batch.shape[0]

def ssim(y, y_pred):
    from skimage.metrics import structural_similarity
    return structural_similarity(y, y_pred, data_range=y.max() - y.min())

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    return math.sqrt(mse(y, y_pred))

#display =======================
def display_img(x, mask, y, y_pred, score=None):
    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,1), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,3), colspan=2)
    ax3 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
    ax1.imshow(x, cmap='gray')
    ax1.set_title('zero-filled')
    ax2.imshow(np.fft.fftshift(mask), cmap='gray')
    ax2.set_title('mask')
    ax3.imshow(y, cmap='gray')
    ax3.set_title('GT')
    ax4.imshow(y_pred, cmap='gray')
    ax4.set_title('reconstruction')
    im5 = ax5.imshow(np.abs(y_pred-y), cmap='gray', vmin=np.abs(y).min(), vmax=np.abs(y).max())
    ax5.set_title('diff')
    fig.colorbar(im5, ax=ax5)
    if score:
        plt.suptitle('score: {:.4f}'.format(score))
    return fig