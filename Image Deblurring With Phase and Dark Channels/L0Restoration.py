import torch
import torch.fft as fft
import torch.nn.functional as F
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from cho_code.opt_fft_size import opt_fft_size
from misc import psf2otf
def L0Restoration(Im, kernel, lambda_, kappa=2.0):
    if not kappa:
        kappa = 2.0
    
    # Get image dimensions
    H, W, D = Im.shape
    sizeI2D = [H, W]
    # Pad image
    #opt_fft_size expects a python list
    # [H + k.shape[0]-1, W + k.shape[0]-1 ]
    Im = wrap_boundary_liu(Im, opt_fft_size([H + kernel.shape[0] - 1, W + kernel.shape[1] - 1 ]))
    
    # Initialize S
    S = Im.clone()
    betamax = 1e5
    fx = torch.tensor([1, -1], dtype=torch.float32)
    fy = torch.tensor([1, -1], dtype=torch.float32)
    N, M, D = Im.shape
    
    # Create otfFx and otfFy
    # otfFx = torch.fft.fftn(fx, s=sizeI2D)
    # otfFy = torch.fft.fftn(fy, s=sizeI2D)
    otfFx = psf2otf(fx, [N,M])
    otfFy = psf2otf(fy, [N,M])
    # Create KER and Den_KER
    # KER = torch.fft.fftn(kernel, s=sizeI2D)
    KER = psf2otf(kernel, [N,M])
    Den_KER = torch.abs(KER)**2
    # Create Denormin2
    Denormin2 = torch.abs(otfFx)**2 + torch.abs(otfFy)**2
    if D > 1:
        Denormin2 = Denormin2.unsqueeze(dim=2).expand(-1, -1, D)
        KER = KER.unsqueeze(dim=2).expand(-1, -1, D)
        Den_KER = Den_KER.unsqueeze(dim=2).expand(-1, -1, D)
    Normin1 = torch.conj(KER.unsqueeze(2)) * fft.fftn(S)
    
    beta = 2 * lambda_
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2
        
        # h = torch.cat([S[:, 1:] - S[:, :-1], S[:, 0:1] - S[:, -1:]], dim=1)
        # v = torch.cat([S[1:, :] - S[:-1, :], S[0:1, :] - S[-1:, :]], dim=0)
        tmph = torch.diff(S,dim=1)
        tmpv = torch.diff(S,dim=0)
        # print(f'tmph shape {tmpv.shape} and {S[S.shape[0]-1:S.shape[0],:,:].shape} abd  ')
        h = torch.cat((tmph,S[:,0:1,:] - S[:,S.shape[1]-1:S.shape[1],:]),dim=1)
        v = torch.cat((tmpv,S[0:1,:,:]-S[S.shape[0]-1:S.shape[0],:,:]))
        # print(f'h and v shapes: {h.shape} , {v.shape}')
        if D == 1:
            t = (h**2 + v**2) < lambda_ / beta
        else:
            t = (h**2 + v**2).sum(dim=2) < lambda_ / beta
            t = t.unsqueeze(dim=2).expand(-1, -1, D)
        
        h[t] = 0
        v[t] = 0
        
        Normin2 = torch.cat([h[:,h.shape[1]-1:h.shape[1],:] - h[:, 0:1,:], -torch.diff(h, dim=1)], dim=1)
        Normin2 += torch.cat([v[v.shape[0]-1:v.shape[0], :,:] - v[0:1,:, :], -torch.diff(v, dim=0)], dim=0)

        FS = (Normin1 + beta * fft.fftn(Normin2)) / Denormin.unsqueeze(2)
        S = fft.ifftn(FS).real
        
        beta *= kappa
    
    # Crop to the original size
    S = S[:H, :W, :]
    
    return S

# Usage
# Replace Im and kernel with your input image and kernel
# Im should be a PyTorch tensor with shape (H, W, D) and kernel should be a PyTorch tensor with shape (kernel_size, kernel_size)
# lambda_ and kappa are hyperparameters you can adjust
# S = L0Restoration(Im, kernel, lambda_, kappa)
