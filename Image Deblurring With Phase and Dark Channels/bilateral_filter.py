import torch
import torch.nn.functional as F
import numpy as np
from skimage import color
from math import ceil
from misc import custompad
from scipy.signal import gaussian
import cv2
def bilateral_filter(img, sigma_s, sigma, boundary_method='replicate', s_size=None):
    if boundary_method is None:
        boundary_method = 'replicate'

    #img = img / 255.0
    
    h, w, d = img.shape

    if d == 3:
        lab = img.clone()
        lab = lab.numpy()
        lab = lab.astype(np.float32)
        lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)
        lab = torch.from_numpy(lab)
        sigma = sigma * 100
    else:
        lab = img.clone()
        sigma = sigma * np.sqrt(d)
    
    if s_size is not None:
        fr = s_size
    else:
        fr = ceil(sigma_s * 3)
    
    torch.set_printoptions(precision=4)
    
    p_img = custompad(img,fr)
    p_lab = custompad(lab,fr)

    u = fr+1
    b = u+h-1
    l = fr+1 
    r = l+w-1

    r_img = torch.zeros((h, w, d), dtype=torch.float)
    w_sum = torch.zeros((h, w), dtype=torch.float)

    # Define the standard deviation (sigma) and kernel size

    # Generate a 1D Gaussian kernel using scipy.signal.gaussian
    gaussian_1d = gaussian(2*fr + 1, std=sigma_s)

    # Create a 2D Gaussian kernel by taking the outer product of the 1D kernel
    spatial_weight = torch.from_numpy(np.outer(gaussian_1d, gaussian_1d))

# Normalize the kernel to ensure the sum of its elements is 1
    spatial_weight /= spatial_weight.sum()

    ss = sigma * sigma
    i = 1
    for y in range(-fr, fr+1):
        for x in range(-fr, fr+1):
            w_s = spatial_weight[y + fr, x + fr]
            n_img = p_img[u + y - 1:b + y , l + x - 1:r + x , :]
            n_lab = p_lab[u + y - 1:b + y , l + x - 1:r + x , :]
            f_diff = lab - n_lab
            f_dist = torch.sum(f_diff ** 2, dim=2)

            w_f = torch.exp(-0.5 * (f_dist / ss))
            w_t = w_s * w_f
            mul = w_t.unsqueeze(2)
            mul = mul.expand(-1,-1,d)
            muld = n_img * mul
            r_img = torch.add(r_img,muld)
            w_sum += w_t
            i = i + 1
    r_img /= w_sum.unsqueeze(2).expand(-1,-1,d)

    return r_img

'''
tensor3d = torch.tensor([
    [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
    ],
    [
        [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
    ]
],dtype=torch.float32)

bf = bilateral_filter(tensor3d,3,0.1)
print(bf)
'''