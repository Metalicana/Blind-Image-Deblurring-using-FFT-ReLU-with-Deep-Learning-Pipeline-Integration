import torch
import torch.nn.functional as F
import numpy as np

def bilateral_filter(img, sigma_s, sigma, boundary_method='replicate', s_size=None):
    if boundary_method is None:
        boundary_method = 'replicate'

    if img.dtype == torch.uint8:
        img = img.float() / 255.0

    h, w, d = img.shape

    if d == 3:
        lab = img.clone()
        sigma = sigma * 100
    else:
        lab = img.clone()
        sigma = sigma * np.sqrt(d)

    if s_size is not None:
        fr = s_size
    else:
        fr = int(np.ceil(sigma_s * 3))

    p_img = F.pad(img, (fr, fr, fr, fr), mode=boundary_method)
    p_lab = F.pad(lab, (fr, fr, fr, fr), mode=boundary_method)

    u, b, l, r = fr, fr + h, fr, fr + w

    r_img = torch.zeros((h, w, d), dtype=img.dtype)
    w_sum = torch.zeros((h, w), dtype=img.dtype)

    spatial_weight = torch.from_numpy(np.float32(np.exp(-0.5 * np.arange(-fr, fr + 1) ** 2 / (sigma_s ** 2))))

    ss = sigma ** 2

    for y in range(-fr, fr + 1):
        for x in range(-fr, fr + 1):
            w_s = spatial_weight[y + fr]
            n_img = p_img[u + y:b + y, l + x:r + x]
            n_lab = p_lab[u + y:b + y, l + x:r + x]
            f_diff = lab - n_lab
            f_dist = torch.sum(f_diff ** 2, dim=2)
            w_f = torch.exp(-0.5 * (f_dist / ss))
            w_t = w_s * w_f
            r_img += n_img * w_t.unsqueeze(2)
            w_sum += w_t

    r_img /= w_sum.unsqueeze(2)

    return r_img
