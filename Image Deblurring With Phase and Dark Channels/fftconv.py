import torch
import torch.fft as fft

def fftconv(I, filt, b_otf=False):
    if I.ndim == 3:
        H, W, ch = I.shape
        otf = torch.fft.fftn(filt, s=(H, W))
        cI = torch.zeros_like(I)
        for i in range(ch):
            cI[..., i] = fftconv(I[..., i], otf, True)
        return cI

    if b_otf:
        cI = torch.ifft(fft.fftn(I) * filt).real
    else:
        cI = torch.ifft(fft.fftn(I) * torch.fft.fftn(filt, s=I.shape)).real

    return cI

# Example usage
#I = torch.randn(256, 256, 3)
#filt = torch.randn(64, 64)
#cI = fftconv(I, filt, True)
#print(cI)
