import torch
import torch.fft as fft
from misc import psf2otf
from dark_channel import dark_channel
from assign_dark_channel_to_pixel import assign_dark_channel_to_pixel
import cv2 as cv
def L0Deblur_dark_channel(Im, kernel, lambda_, wei_grad, kappa=2.0):
    S = Im.clone()
    betamax = 1e5
    fx = torch.tensor([1, -1])
    fy = fx.transpose(0,1)
    N, M, D = Im.shape
    sizeI2D = [N, M]
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = torch.abs(KER)**2

    Denormin2 = torch.abs(otfFx)**2 + torch.abs(otfFy)**2
    if D > 1:
        Denormin2 = Denormin2.unsqueeze(2).expand(-1, -1, D)
        KER = KER.unsqueeze(2).expand(-1, -1, D)
        Den_KER = Den_KER.unsqueeze(2).expand(-1, -1, D)
    
    Normin1 = torch.conj(KER) * fft.fftn(S)

    dark_r = 35
    ret2,th2 = cv.threshold(S*S,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    mybeta_pixel = lambda_ / ret2
    maxbeta_pixel = 2**3

    while mybeta_pixel < maxbeta_pixel:
        J, J_idx = dark_channel(S, dark_r)
        u = J.clone()
        if D == 1:
            t = u**2 < lambda_ / mybeta_pixel
        else:
            #t = u.norm(dim=2)**2 < lambda_ / mybeta_pixel
            t = torch.sum(u*u, dim=2) < lambda_/mybeta_pixel
            t = t.unsqueeze(2).expand(-1, -1, D)
        u[t] = 0

        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        beta = 2 * wei_grad
        while beta < betamax:
            Denormin = Den_KER + beta * Denormin2 + mybeta_pixel

            tmph = torch.diff(S,dim=1)
            tmpv = torch.diff(S,dim=0)
            h = torch.cat((tmph,S[:,0,:] - S[:,S.shape[1]-1:S.shape[1],:]),dim=1)
            v = torch.cat((tmpv,S[0,:,:]-S[S.shape[0]-1:S.shape[0],:,:]))

            if D == 1:
                t = (h**2 + v**2) < wei_grad / beta
            else:
                t = (h**2 + v**2).sum(dim=2) < wei_grad / beta
                t = t.unsqueeze(dim=2).expand(-1, -1, D)

            h[t] = 0
            v[t] = 0

            Normin2 = torch.cat([h[:, -1:] - h[:, :-1], -torch.diff(h, dim=1)], dim=1)
            Normin2 += torch.cat([v[-1:, :] - v[:-1, :], -torch.diff(v, dim=0)], dim=0)

            FS = (Normin1 + beta * fft.fftn(Normin2) + mybeta_pixel * fft.fftn(u)) / Denormin
            S = fft.ifftn(FS).real
            beta = beta * kappa
            if wei_grad == 0:
                break

        mybeta_pixel = mybeta_pixel * kappa

    return S


# Example usage
#Im = torch.randn(256, 256, 3)
#kernel = torch.randn(64, 64)
#lambda_ = 0.001
#wei_grad = 0.001
#kappa = 2.0
#S = L0Deblur_dark_channel(Im, kernel, lambda_, wei_grad, kappa)





