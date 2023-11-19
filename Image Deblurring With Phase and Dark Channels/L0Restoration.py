import torch
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from cho_code.opt_fft_size import opt_fft_size
from misc import psf2otf, fft, fft2, ifft, ifft2
def L0Restoration(Im, kernel, lambda_, kappa=2.0):
    if not kappa:
        kappa = 2.0
    
    # Get image dimensions
    H, W, D = Im.shape
    # Pad image
    #opt_fft_size expects a python list
    # [H + k.shape[0]-1, W + k.shape[0]-1 ]

    #THIS COMMENT IS ONLY FOR TESTING PUT IT BACK TO HOW IT WAS
    Im = wrap_boundary_liu(Im, opt_fft_size([H + kernel.shape[0] - 1, W + kernel.shape[1] - 1 ]))
    # print('im here')
    # print(Im.squeeze()[0:5,0:5])
    # Initialize S
    S = Im.clone()
    # print(S.shape)
    betamax = 1e5
    fx = torch.tensor([[1, -1]], dtype=torch.float32)
    fy = fx.t()
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
        Normin1 = torch.conj(KER) * fft2(S)
    else:
        
    # print(f'Kernel dims {KER.shape}, S shape {S.shape}')
        Normin1 = torch.conj(KER).unsqueeze(-1).expand_as(S) * fft2(S)
    # print(fft2(S).squeeze())
    beta = 2 * lambda_
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2
        
        # h = torch.cat([S[:, 1:] - S[:, :-1], S[:, 0:1] - S[:, -1:]], dim=1)
        # v = torch.cat([S[1:, :] - S[:-1, :], S[0:1, :] - S[-1:, :]], dim=0)
        tmph = torch.diff(S,dim=1)
        tmpv = torch.diff(S,dim=0)
        
        tmph2 = S[:,0,:] - S[:,-1,:]
        
        tmph2 = tmph2.unsqueeze(1)
        
        #print(tmph.shape, (S[:,0,:] - S[:,S.shape[1]-1:S.shape[1],:]).shape)
        h = torch.cat((tmph,tmph2),dim = 1)
        v = torch.cat((tmpv,S[0,:,:]-S[-1:,:,:]))
        if D == 1:
            t = (h**2 + v**2) < lambda_ / beta
        else:
            t = (h**2 + v**2).sum(dim=2) < lambda_ / beta
            
            if t.ndim!=3 and t.shape[t.ndim-1]!=D:
                t = t.unsqueeze(dim=2).expand(-1, -1, D)
            
        if D == 3: 
            h = h.view(t.shape[0],t.shape[1],t.shape[2])
        h[t] = 0
        v[t] = 0
        h1 = h[:, -1, :] - h[:,0,:]
        h1 = h1.unsqueeze(1)
        h2 = -torch.diff(h,dim=1)
        Normin2 = torch.cat((h1,h2), dim=1)
        v1 = v[-1,:,:] - v[0,:,:]
        v2 = -torch.diff(v,dim=0)
        v1 = v1.unsqueeze(0)
        Normin2 += torch.cat((v1,v2))
        if D == 1:
            FS = (Normin1 + beta * fft2(Normin2)) / Denormin.unsqueeze(-1).expand_as(Normin1)
        else:
            FS = (Normin1 + beta * fft2(Normin2)) / Denormin
        S = ifft2(FS).real
       
        beta *= kappa
        
    # Crop to the original size
    S = S[:H, :W, :]
    
    return S
# Usage
# Replace Im and kernel with your input image and kernel
# Im should be a PyTorch tensor with shape (H, W, D) and kernel should be a PyTorch tensor with shape (kernel_size, kernel_size)
# lambda_ and kappa are hyperparameters you can adjust
# S = L0Restoration(Im, kernel, lambda_, kappa)
