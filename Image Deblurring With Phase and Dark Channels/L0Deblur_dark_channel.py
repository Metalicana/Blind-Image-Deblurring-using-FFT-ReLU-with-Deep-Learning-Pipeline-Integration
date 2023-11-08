import torch
import torch.fft as fft
import math

def L0Deblur_dark_channel(Im, kernel, lambda_, wei_grad, kappa=2.0):
    if kappa is None:
        kappa = 2.0

    S = Im.clone()
    betamax = 1e5
    fx = torch.tensor([1, -1])
    fy = torch.tensor([1, -1]).view(-1, 1)
    N, M, D = Im.shape
    sizeI2D = (N, M)
    otfFx = fft.fftn(fx, s=sizeI2D)
    otfFy = fft.fftn(fy, s=sizeI2D)

    KER = fft.fftn(kernel, s=sizeI2D)
    Den_KER = torch.abs(KER)**2

    Denormin2 = torch.abs(otfFx)**2 + torch.abs(otfFy)**2
    if D > 1:
        Denormin2 = Denormin2.repeat(1, 1, D)
        KER = KER.repeat(1, 1, D)
        Den_KER = Den_KER.repeat(1, 1, D)
    
    Normin1 = torch.conj(KER) * fft.fftn(S, s=sizeI2D)

    dark_r = 35
    mybeta_pixel = lambda_ / (torch.tensor(S).var().sqrt().item() + 1e-6)
    maxbeta_pixel = 2**3

    while mybeta_pixel < maxbeta_pixel:
        J, J_idx = dark_channel(S, dark_r)
        u = J.clone()
        if D == 1:
            t = u**2 < lambda_ / mybeta_pixel
        else:
            t = u.norm(dim=2)**2 < lambda_ / mybeta_pixel
            t = t.repeat(1, 1, D)
        u[t] = 0

        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        beta = 2 * wei_grad
        while beta < betamax:
            Denormin = Den_KER + beta * Denormin2 + mybeta_pixel

            h = torch.cat((S[:, :, -1].unsqueeze(2) - S[:, :, 0].unsqueeze(2), S[:, :, :-1] - S[:, :, 1:]), dim=2)
            v = torch.cat((S[-1, :, :].unsqueeze(0) - S[0, :, :].unsqueeze(0), S[:-1, :, :] - S[1:, :, :]), dim=0)

            if D == 1:
                t = (h**2 + v**2) < wei_grad / beta
            else:
                t = (h.norm(dim=2)**2 + v.norm(dim=2)**2) < wei_grad / beta
                t = t.repeat(1, 1, D)

            h[t] = 0
            v[t] = 0

            Normin2 = torch.cat((h[:, -1, :].unsqueeze(1) - h[:, 0, :].unsqueeze(1), -torch.diff(h, dim=1)), dim=1)
            Normin2 = Normin2 + torch.cat((v[-1, :, :].unsqueeze(0) - v[0, :, :].unsqueeze(0), -torch.diff(v, dim=0)), dim=0)

            FS = (Normin1 + beta * fft.ifftn(Normin2) + mybeta_pixel * fft.fftn(u)) / Denormin
            S = FS.real
            beta = beta * kappa
            if wei_grad == 0:
                break

        mybeta_pixel = mybeta_pixel * kappa

    return S

def dark_channel(I, dark_r):
    J = I.clone()
    J_idx = torch.zeros_like(J, dtype=torch.long)

    for i in range(dark_r, I.shape[0] - dark_r):
        for j in range(dark_r, I.shape[1] - dark_r):
            patch = I[i - dark_r:i + dark_r + 1, j - dark_r:j + dark_r + 1, :]
            min_channel = torch.min(patch, dim=(0, 1))
            min_value, min_idx = torch.min(min_channel, dim=0)

            J[i, j, :] = min_channel[min_idx]
            J_idx[i, j, :] = min_idx

    return J, J_idx

def assign_dark_channel_to_pixel(I, dark_channel, dark_channel_idx, dark_r):
    J = I.clone()

    for i in range(dark_r, I.shape[0] - dark_r):
        for j in range(dark_r, I.shape[1] - dark_r):
            idx = dark_channel_idx[i, j, :]
            J[i, j, :] = dark_channel[i, j, idx]

    return J

# Example usage
#Im = torch.randn(256, 256, 3)
#kernel = torch.randn(64, 64)
#lambda_ = 0.001
#wei_grad = 0.001
#kappa = 2.0
#S = L0Deblur_dark_channel(Im, kernel, lambda_, wei_grad, kappa)





