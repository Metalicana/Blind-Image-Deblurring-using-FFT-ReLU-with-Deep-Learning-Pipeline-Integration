import torch
import torch.fft as fft
import torch.nn.functional as F
from cho_code import wrap_boundary_liu, opt_fft_size
import bilateral_filter
import deblurring_adm_aniso
from L0Restoration import L0Restoration
def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    H, W, _ = y.shape
    y_pad = wrap_boundary_liu(y, opt_fft_size([H, W] + list(kernel.shape)) - 1)
    Latent_tv = []
    
    for c in range(y.shape[2]):
        Latent_tv.append(deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1))
    
    Latent_tv = torch.stack(Latent_tv, dim=2)
    Latent_tv = Latent_tv[:H, :W, :]
    
    if weight_ring == 0:
        return Latent_tv
    
    Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W, :]
    
    diff = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff, 3, 0.1)
    result = Latent_tv - weight_ring * bf_diff
    
    return result

# Usage
# Replace y and kernel with your input image and kernel
# y should be a PyTorch tensor with shape (H, W, C) and kernel should be a PyTorch tensor with shape (kernel_size, kernel_size)
# lambda_tv, lambda_l0, and weight_ring are hyperparameters you can adjust
# result = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
