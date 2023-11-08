import torch
import torch.fft as fft

def deblurring_adm_aniso(B, k, lambda_val, alpha):
    beta = 1 / lambda_val
    beta_rate = 2 * 2**0.5
    beta_min = 0.001

    m, n = B.shape
    I = B.clone()

    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("Blur kernel k must be odd-sized.")

    def computeDenominator(y, k):
        otfk = torch.fft.fftn(k, s=y.shape)
        Nomin1 = torch.conj(otfk) * torch.fft.fftn(y)
        Denom1 = torch.abs(otfk)**2
        Denom2 = torch.abs(torch.fft.fftn(torch.tensor([1, -1], dtype=y.dtype, device=y.device), s=y.shape))**2 + \
                 torch.abs(torch.fft.fftn(torch.tensor([1, -1], dtype=y.dtype, device=y.device).view(-1, 1), s=y.shape))**2
        return Nomin1, Denom1, Denom2

    Ix = torch.cat([I[:, 1:] - I[:, :-1], I[:, 0].unsqueeze(1) - I[:, -1].unsqueeze(1)], dim=1)
    Iy = torch.cat([I[1:] - I[:-1], (I[0] - I[-1]).unsqueeze(0)], dim=0)

    while beta > beta_min:
        gamma = 1 / (2 * beta)
        Nomin1, Denom1, Denom2 = computeDenominator(B, k)

        if alpha == 1:
            Wx = torch.max(torch.abs(Ix) - beta * lambda_val, torch.tensor(0, dtype=Ix.dtype, device=Ix.device)) * torch.sign(Ix)
            Wy = torch.max(torch.abs(Iy) - beta * lambda_val, torch.tensor(0, dtype=Iy.dtype, device=Iy.device)) * torch.sign(Iy)
        else:
            Wx = solve_image(Ix, 1 / (beta * lambda_val), alpha)
            Wy = solve_image(Iy, 1 / (beta * lambda_val), alpha)

        Wxx = torch.cat([(Wx[:, -1] - Wx[:, 0]).unsqueeze(1), -torch.cat([Wx[:, 1:] - Wx[:, :-1], Wx[:, 0].unsqueeze(1) - Wx[:, -1].unsqueeze(1)], dim=1)], dim=1)
        Wxx += torch.cat([(Wy[-1, :] - Wy[0, :]).unsqueeze(0), -torch.cat([Wy[1:] - Wy[:-1], (Wy[0, :] - Wy[-1, :]).unsqueeze(0)], dim=0)], dim=0)

        Fyout = (Nomin1 + gamma * fft.fftn(Wxx)) / Denom1
        I = torch.real(fft.ifftn(Fyout))
        Ix = torch.cat([I[:, 1:] - I[:, :-1], I[:, 0].unsqueeze(1) - I[:, -1].unsqueeze(1)], dim=1)
        Iy = torch.cat([I[1:] - I[:-1], (I[0] - I[-1]).unsqueeze(0)], dim=0)
        beta = beta / beta_rate

    return I

def solve_image(data, tau, alpha):
    return torch.sign(data) * torch.relu(torch.abs(data) - tau) ** (1 / alpha)

# Example usage:
# B, k, lambda_val, alpha = ...  # Set your inputs here
# deblurred_image = deblurring_adm_aniso(B, k, lambda_val, alpha)
