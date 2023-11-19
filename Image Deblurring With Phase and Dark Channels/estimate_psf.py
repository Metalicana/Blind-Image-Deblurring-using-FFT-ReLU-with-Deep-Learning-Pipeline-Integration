import torch
import torch.fft as fft
import torch.nn.functional as F
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg
from misc import psf2otf, otf2psf, fft2, ifft2
from cho_code.conjgrad import conjgrad
def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size):
    
    
    # import numpy as np
    # nparr = latent_x.squeeze().numpy()
    # np.savetxt('output.txt', nparr)
    # from misc import visualize_image
    # visualize_image(latent_x.squeeze())
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)
    # print(f'Latent XF {latent_xf.shape} & blurred XF {blurred_xf.shape}')
    b_f = torch.conj(latent_xf) * blurred_xf + torch.conj(latent_yf) * blurred_yf
    # print(b_f.shape)
    b = otf2psf(b_f.squeeze(), torch.tensor(psf_size).tolist())
    # print(f'b inside estimate psf {b.shape}')
    p_m = torch.conj(latent_xf) * latent_xf + torch.conj(latent_yf) * latent_yf
    p_img_size = torch.tensor(blurred_xf.shape)
    p_psf_size = torch.tensor(psf_size)
    p_lambda = weight

    psf = torch.ones(psf_size) / torch.prod(torch.tensor(psf_size), dtype=torch.float32)
    dict = {
        'img_size': p_img_size,
        'm' : p_m,
        'psf_size': p_psf_size,
        'lambda' : p_lambda
    }
    # psf = conjgrad(psf, b, 20, 1e-5, compute_Ax, (p_m, p_img_size, p_psf_size, p_lambda))
    psf = conjgrad(psf, b, 20, 1e-5, compute_Ax, dict)
    psf[psf < torch.max(psf) * 0.05] = 0
    psf /= torch.sum(psf)
    
    return psf

def compute_Ax(x, p):
    
    x_f = psf2otf(x, p['img_size'].tolist())

    y = otf2psf(p['m'].squeeze()*x_f.squeeze(), p['psf_size'].tolist())
    y = y + p['lambda']*x
    # x_f = fft.ifftshift(x)
    # x_f = fft.fftn(x_f, s=p[1])
    # y = fft.ifft2(p[0] * x_f).real
    # y += p[3] * x
    # y = fft.fftshift(y)
    return y

# def conjgrad(x, b, niter, tol, A, A_args=()):
#     x = x.clone()
#     b = b.clone()
#     print(f'b shape: {b.shape}')
#     x = A(x, A_args)
#     print(f'x shape: {x.shape}')
#     r = b - A(x, A_args)
#     p = r.clone()
#     rsold = torch.sum(r * r)
#     for i in range(niter):
#         Ap = A(p, A_args)
#         alpha = rsold / torch.sum(p * Ap)
#         x += alpha * p
#         r -= alpha * Ap
#         rsnew = torch.sum(r * r)
#         if torch.sqrt(rsnew) < tol:
#             break
#         p = r + (rsnew / rsold) * p
#         rsold = rsnew
#     return x

# Example usage
# blurred_x = torch.randn(128, 128)
# blurred_y = torch.randn(128, 128)
# latent_x = torch.randn(128, 128)
# latent_y = torch.randn(128, 128)
# weight = 0.01
# psf_size = (64, 64)

# psf = estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size)
# print(psf)
