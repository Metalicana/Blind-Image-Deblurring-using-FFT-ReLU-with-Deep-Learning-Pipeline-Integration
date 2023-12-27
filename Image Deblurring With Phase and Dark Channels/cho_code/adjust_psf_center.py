import torch
import torch.nn.functional as F

def adjust_psf_center(psf):
    # print(psf)
    psf_height, psf_width = psf.shape
    Y, X = torch.meshgrid(torch.arange(1,psf_height+1), torch.arange(1,psf_width+1))
    xc1 = torch.sum(psf * X).item()
    yc1 = torch.sum(psf * Y).item()
    xc2 = (float(psf_width)+1.0)/2.0
    yc2 = (float(psf_height)+1.0)/2.0
    
    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)
    # print(xc1, yc1, xc2, yc2, xshift, yshift)
    # print(torch.tensor([[1,0,-xshift],[0,1,-yshift]],dtype=torch.float32))
    A_psf = torch.tensor([[1,0,-xshift],[0,1,-yshift]],dtype=torch.float32)
    psf = warp_image(psf, A_psf)
    return psf

def warp_projective2(im, A):
    if A.size(0) > 2:
        A = A[:2,:]
    x, y = torch.meshgrid(torch.arange(1, im.shape[0] + 1), torch.arange(1, im.shape[1] + 1))

    x = x.squeeze()
    # xmx = x.max()
    # xmn = x.min()
    # x = (2*(x-xmn)/(xmx-xmn))-1
    y = y.squeeze()
    # ymx = y.max()
    # ymn = y.min()
    # y = (2*(y-ymn)/(ymx-ymn))-1

    # grid = torch.stack((y,x), dim=-1)
    # grid = grid.unsqueeze(0)
    # print(x.flatten())
    coords = torch.stack([x.flatten(), y.flatten()], dim=0)
    #Might need to change the order of the x, y 
    # print(coords.shape)
    # homogeneous_coords = torch.cat([coords, torch.ones(coords.shape[0], coords.shape[1])], dim=0)
    #homogeneous_coords will be a 3xN tensor which 1s in the last row
    homogeneous_coords = torch.cat([coords, torch.ones(1, coords.shape[1])], dim=0)
    # print(homogeneous_coords)
    warped_coords = torch.mm(A, homogeneous_coords)

    x_prime = warped_coords[0, :].unsqueeze(0)
    
    
    y_prime = warped_coords[1, :].unsqueeze(0)
    
    x_prime = x_prime - 1
    y_prime = y_prime - 1

    # print(y_prime)
    valid_mask = (x_prime >= 0) & (x_prime < int(im.shape[0]) ) & (y_prime >= 0) & (y_prime < int(im.shape[1]))
    
    result = torch.zeros_like(x_prime,dtype=im.dtype)

    result[valid_mask] = im[y_prime[valid_mask].long(), x_prime[valid_mask].long() ]

    result = result.squeeze()
    return result.reshape(im.shape).t()
    # return result


def warp_image(img, M):
    
    if img.dim() == 3 and img.size(0) == 3:
        warped = torch.zeros_like(img)
        for channel in range(3):
            warped[channel] = warp_projective2(img[channel], M)
    else:
        warped = warp_projective2(img, M)
    warped[torch.isnan(warped)] = 0
    return warped