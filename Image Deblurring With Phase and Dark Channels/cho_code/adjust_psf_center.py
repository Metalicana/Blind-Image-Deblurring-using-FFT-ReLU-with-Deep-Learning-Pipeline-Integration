import torch
import torch.nn.functional as F

def adjust_psf_center(psf):
    psf_height, psf_width = psf.shape
    Y, X = torch.meshgrid(torch.arange(1,psf_height+1), torch.arange(1,psf_width+1))
    xc1 = torch.sum(psf * X)
    yc1 = torch.sum(psf * Y)
    xc2 = (psf_width+1)/2
    yc2 = (psf_height)/2
    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)
    A_psf = torch.tensor([1,0,-xshift],[0,1,-yshift],dtype=torch.float32)
    psf = warp_image(psf, A_psf)
    return psf

def warp_projective2(im, A):
    if A.size(0) > 2:
        A = A[:2,:]
    x, y = torch.meshgrid(torch.arange(1, im.shape[1] + 1), torch.arange(1, im.shape[0] + 1))
    coords = torch.stack([x.view(-1), y.view(-1)], dim=0)
    homogeneous_coords = torch.cat([coords, torch.ones(1, coords.shape[1])], dim=0)
    warped_coords = torch.mm(A, homogeneous_coords)
    x_prime = warped_coords[0, :] / warped_coords[2, :]
    y_prime = warped_coords[1, :] / warped_coords[2, :]
    result = F.grid_sample(im.unsqueeze(0).unsqueeze(0), torch.stack([x_prime, y_prime], dim=0).permute(1, 0, 2).unsqueeze(0), mode='bilinear')
    result = result.squeeze(0).squeeze(0)
    return result
def warp_image(img, M):
    if img.dim() == 3 and img.size(0) == 3:
        warped = torch.zeros_like(img)
        for channel in range(3):
            warped[channel] = warp_projective2(img[channel], M)
    else:
        warped = warp_projective2(img[channel], M)
    warped[torch.isnan(warped)] = 0
    return warped