import torch
import torch.fft as fft
import torch.nn.functional as F
from cho_code.opt_fft_size import opt_fft_size
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from bilateral_filter import bilateral_filter
from deblurring_adm_aniso import deblurring_adm_aniso
from L0Restoration import L0Restoration
from L0Deblur_FTR import L0Deblur_FTR
def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    # print(kernel)
    H, W, C = y.shape
    dim_list = [H, W]
    param = [x + y - 1 for x, y in zip(dim_list, list(kernel.shape))]
    p = opt_fft_size(param)
    y_pad = wrap_boundary_liu(y, p)
    # print(f'y_pad shape is {y_pad.shape}')
    Latent_tv = torch.Tensor().type(torch.float32)
    
    for c in range(C):
        aniso = deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1)
        n = aniso.ndim
        aniso = aniso.unsqueeze(n)
        Latent_tv = torch.cat((Latent_tv,aniso),dim = n)
    # print(Latent_tv.shape)
    # print("TV")
    # print(Latent_tv.shape)
    # print(Latent_tv)
    Latent_tv = Latent_tv[:H, :W, :]
    
    if weight_ring == 0:
        return Latent_tv
    
    Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W, :]
    # print(Latent_l0.shape)
    diff = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff, 3, 0.1)
    result = Latent_tv - weight_ring * bf_diff
    # print(result.shape)
    return result

# Usage
# Replace y and kernel with your input image and kernel
# y should be a PyTorch tensor with shape (H, W, C) and kernel should be a PyTorch tensor with shape (kernel_size, kernel_size)
# lambda_tv, lambda_l0, and weight_ring are hyperparameters you can adjust
# result = _artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)

img = torch.arange(1, 250*250*3 + 1).reshape((250, 250, 3))
ker = torch.arange(1, 25*25 + 1).reshape(25,25)

