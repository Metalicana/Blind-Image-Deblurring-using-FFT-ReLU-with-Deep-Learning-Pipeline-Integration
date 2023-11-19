import torch
import torch.nn.functional as F
import numpy as np
import math
from misc import conv2
def threshold_pxpy_v1(latent, psf_size, threshold=None):
    # Check if threshold is provided
    if threshold is None:
        threshold = 0
        b_estimate_threshold = True
    else:
        b_estimate_threshold = False

    # Denoised is the same as latent in this version
    denoised = latent.clone()
    # print(f'denoised shape {denoised.shape}')
    # Define derivative filters
    dx = torch.tensor([[-1, 1], [0, 0]], dtype=torch.float)
    dy = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float)

    # Compute gradients
    
    px = conv2(denoised, dx, 'valid')
    py = conv2(denoised, dy, 'valid')
    pm = px**2 + py**2
    # print(pm[0:10,0:10].squeeze())
    # Estimate threshold if necessary
    if b_estimate_threshold:
        pd = torch.atan(py/ px)
        pm_steps = torch.arange(0, 2, 0.00006)

        pm_pos1 = pm[(pd >= 0) & (pd < (math.pi / 4.0))]
        pm_pos2 = pm[(pd >= (math.pi / 4.0)) & (pd < (math.pi / 2.0))]
        pm_neg1 = pm[(pd >= (-math.pi / 4.0)) & (pd < 0)]
        pm_neg2 = pm[(pd >= (-math.pi / 2.0)) & (pd < (-math.pi / 4.0))]

        H1 = torch.cumsum(torch.flip(torch.histc(pm_pos1, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
        H2 = torch.cumsum(torch.flip(torch.histc(pm_pos2, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
        H3 = torch.cumsum(torch.flip(torch.histc(pm_neg1,pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
        H4 = torch.cumsum(torch.flip(torch.histc(pm_neg2, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)

        th = psf_size.max() * 20  # Adjust this as needed
        if th <= 10:
            th = 10

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            # print(min_h)
            if min_h >= th:
                threshold = pm_steps[-t]
                break

    # Thresholding
    # print(threshold)
    m = pm < threshold
    while torch.all(m == 1):
        threshold = threshold * 0.81
        m = pm < threshold
    px[m] = 0
    py[m] = 0

    # Modify threshold based on the condition
    
    if b_estimate_threshold:
        threshold = threshold
    else:
        threshold = threshold / 1.1
    # print(f'Threshold pxpy {threshold}')
    return px, py, threshold