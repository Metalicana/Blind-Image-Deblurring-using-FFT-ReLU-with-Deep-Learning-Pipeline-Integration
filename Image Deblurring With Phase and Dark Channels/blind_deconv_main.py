import torch
import torch.nn.functional as F

from L0Restoration import L0Restoration
from L0Deblur_FTR import L0Deblur_FTR
from scipy.ndimage import label
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from cho_code.threshold_pxpy_v1 import threshold_pxpy_v1
from cho_code.opt_fft_size import opt_fft_size
from L0Deblur_dark_channel import L0Deblur_dark_channel
from estimate_psf import estimate_psf
from misc import conv2
from skimage import measure
#expect to return a dictionary,
#Where NumObjects wil refer to how many different connected components there are
#Each array of CC["pixelIdxList"] will contain list of indices of the that particular connected component
# So, CC["pixelIdxList"] is a 2D array, where CC["pixelIdxList"][I][J] referes to the Ith connected components'
# Jth index (the indices are of course 2D tuple (X, Y ) value)

def connected_components(bw):
    CC = {}
    t = bw.clone()
    t = t.numpy()
    t[t>0] = 1
    lbl = measure.label(t)
    lbl = torch.from_numpy(lbl)
    num = torch.max(lbl)
    CC['NumObjects'] = num
    CC['PixelIdxList'] = []
    for n in range(1, num+1):
        indices = torch.nonzero(lbl == n).tolist()
        CC['PixelIdxList'].append(indices)

    return CC

def blind_deconv_main(blur_B, k, lambda_ftr, lambda_dark, lambda_grad, threshold, opts):
    # Derivative filters
    dx = torch.Tensor([[-1, 1], [0, 0]])
    dy = torch.Tensor([[-1, 0], [1, 0]])

    H, W, _ = blur_B.size()
    
    # Wrap boundary for convolution
    # blur_B_w = F.conv2d(blur_B.permute(2, 0, 1).unsqueeze(0), k.permute(2, 3, 0, 1)).squeeze(0).permute(1, 2, 0)
    # print(f'blurB shape: {blur_B.shape}')

    #THIS COMMENTED LINE IS ABSOLUTELY CRUCIAL 
    blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size( [H + k.shape[0]-1, W + k.shape[0]-1 ] ))
    # print(blur_B_w[0:10,0:10].squeeze())
    blur_B_tmp = blur_B[0:H,0:W,:]
    Bx = conv2(blur_B_tmp, dx, 'valid')
    By = conv2(blur_B_tmp, dy, 'valid')
    # print(Bx[0:10,0:10].squeeze())
    # Bx = F.conv2d(blur_B_w.permute(2, 0, 1).unsqueeze(0), dx.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    # By = F.conv2d(blur_B_w.permute(2, 0, 1).unsqueeze(0), dy.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    
    for iter in range(1, opts['xk_iter'] + 1):
        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[0:H, 0:W, :]
        else:
            # print(f'before L0 restoration S shape {blur_B.shape}')
            # print(blur_B[0:10,0:10].squeeze())
            # print(k)
            # print(lambda_grad)
            # S = L0Restoration(blur_B,k, lambda_grad,2.0)
            S = L0Deblur_FTR(blur_B, k, lambda_ftr, lambda_grad, 2.0)
            # from misc import visualize_image
            # visualize_image(S.squeeze())

            # print(S[0:10,0:10].squeeze())
            # print(f'After L0 restoration S shape {S.shape}')
        
        latent_x, latent_y, threshold = threshold_pxpy_v1(S, max(k.size()), threshold)
        k_prev = k.clone()
        
        # Estimate PSF (kernel)
        #k.size() is list ?
        # print(k)
        # print(k_prev.size())
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.size())
        
        # Prune isolated noise in the kernel
        # print('printing kernel after L0Restoration and estimate PSF')
        # print(k)
        print('pruning isolated noise in kernel...')
        # print(k)
        CC = connected_components(k)
        for ii in range(1, CC['NumObjects'] + 1):
            idx = torch.tensor(CC['PixelIdxList'][ii-1])
            currsum = k[idx[:,0], idx[:,1]].sum()
            if currsum < 0.1:
                k[idx[:,0],idx[:,1]] = 0
        
        k[k < 0] = 0
        k /= torch.sum(k)
        
        # Parameter updating
        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        else:
            lambda_dark = 0
        
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)
        else:
            lambda_grad = 0
        if lambda_ftr != 0:
            lambda_ftr = max(lambda_ftr / 1.1, 1e-4)
        #git hub aint working
        # Visualization (you may need to modify this part)
        # To display images in Python, you can use libraries like Matplotlib
        # For saving images, you can use PIL (Pillow) library
        # For now, this part is commented as it depends on your specific setup
        """
        import matplotlib.pyplot as plt
        S[S < 0] = 0
        S[S > 1] = 1
        plt.subplot(1, 3, 1)
        plt.imshow(blur_B)
        plt.title('Blurred image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(S)
        plt.title('Interim latent image')
        
        plt.subplot(1, 3, 3)
        plt.imshow(k)
        plt.title('Estimated kernel')
        
        plt.show()
        """
    
    k[k < 0] = 0
    k /= torch.sum(k)
    
    return k, lambda_ftr,lambda_dark, lambda_grad, S

# You'll need to implement or import the missing functions (e.g., L0Deblur_dark_chanel, L0Restoration, estimate_psf,
# threshold_pxpy_v1, and connected_components) as well as set up the visualization and image saving part.
