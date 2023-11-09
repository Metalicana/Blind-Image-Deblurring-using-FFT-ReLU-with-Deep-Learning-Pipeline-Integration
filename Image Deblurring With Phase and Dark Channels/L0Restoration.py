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
    #print(S.shape)
    betamax = 1e5
    fx = torch.tensor([1, -1], dtype=torch.float32)
    fy = torch.tensor([1, -1], dtype=torch.float32)
    N, M, D = Im.shape
    
    # Create otfFx and otfFy
    # otfFx = torch.fft.fftn(fx, s=sizeI2D)
    # otfFy = torch.fft.fftn(fy, s=sizeI2D)
    otfFx = psf2otf(fx, output_shape=(N,M))
    otfFy = psf2otf(fy, output_shape=(N,M))
    # Create KER and Den_KER
    # KER = torch.fft.fftn(kernel, s=sizeI2D)
    KER = psf2otf(kernel, output_shape=(N,M))
    Den_KER = torch.abs(KER)**2
    
    # Create Denormin2
    Denormin2 = torch.abs(otfFx)**2 + torch.abs(otfFy)**2
    if D > 1:
        Denormin2 = Denormin2.unsqueeze(dim=2).expand(-1, -1, D)
        KER = KER.unsqueeze(dim=2).expand(-1, -1, D)
        Den_KER = Den_KER.unsqueeze(dim=2).expand(-1, -1, D)
    #print(f'Kernel dims {KER.shape}, S shape {S.shape}')
    Normin1 = torch.conj(KER) * fft.fftn(S)
    
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
        FS = (Normin1 + beta * fft.fftn(Normin2)) / Denormin
        S = fft.ifftn(FS).real
       
        beta *= kappa
        
    # Crop to the original size
    S = S[:H, :W, :]
    
    return S
def psf2otf(psf, output_shape):
    # Check if psf is a 1D or 2D tensor
    if len(psf.shape) == 1:
        # If psf is a 1D vector, reshape it to a 2D matrix
        psf = psf.view(1, -1)
    
    # Calculate the center of the PSF
    psf_center = torch.tensor(psf.shape) // 2

    # Pad the PSF to match the desired output shape
    psf_padded = torch.nn.functional.pad(psf, (0, output_shape[1] - psf.shape[1], 0, output_shape[0] - psf.shape[0]))

    # Shift the padded PSF to the center
    psf_shifted = torch.roll(psf_padded, shifts=tuple(psf_center), dims=(1, 0))

    # Compute the Fourier Transform of the shifted PSF
    otf = fft.fftn(psf_shifted)

    return otf
# Usage
# Replace Im and kernel with your input image and kernel
# Im should be a PyTorch tensor with shape (H, W, D) and kernel should be a PyTorch tensor with shape (kernel_size, kernel_size)
# lambda_ and kappa are hyperparameters you can adjust
# S = L0Restoration(Im, kernel, lambda_, kappa)
