import torch
import torch.nn.functional as F
from cho_code import threshold_pxpy_v1
import estimate_psf
from L0Restoration import L0Restoration
from scipy.ndimage import label
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from cho_code.threshold_pxpy_v1 import threshold_pxpy_v1
from cho_code.opt_fft_size import opt_fft_size
from L0Deblur_dark_channel import L0Deblur_dark_channel
from estimate_psf import estimate_psf
def connected_components(bw):
    labeled_image, num_features = label(bw)
    component_list = []
    
    for label_id in range(1, num_features + 1):
        component = (labeled_image == label_id)
        component_list.append(component)
    
    CC = {}
    CC["NumObjects"] = num_features
    CC["PixelIdxList"] = component_list

    return CC

def blind_deconv_main(blur_B, k, lambda_dark, lambda_grad, threshold, opts):
    # Derivative filters
    dx = torch.Tensor([[-1, 1], [0, 0]])
    dy = torch.Tensor([[-1, 0], [1, 0]])

    H, W, _ = blur_B.size()
    
    # Wrap boundary for convolution
    # blur_B_w = F.conv2d(blur_B.permute(2, 0, 1).unsqueeze(0), k.permute(2, 3, 0, 1)).squeeze(0).permute(1, 2, 0)
    # print(f'blurB shape: {blur_B.shape}')
    blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size( [H + k.shape[0]-1, W + k.shape[0]-1 ] ))
    Bx = F.conv2d(blur_B_w.permute(2, 0, 1).unsqueeze(0), dx.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    By = F.conv2d(blur_B_w.permute(2, 0, 1).unsqueeze(0), dy.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    
    for iter in range(1, opts['xk_iter'] + 1):
        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[0:H, 0:W, :]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        
        latent_x, latent_y, threshold = threshold_pxpy_v1(S, max(k.size()), threshold)
        
        k_prev = k.clone()
        
        # Estimate PSF (kernel)
        
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.size())
        
        # Prune isolated noise in the kernel
        CC = connected_components(k)
        for ii in range(1, CC['NumObjects'] + 1):
            currsum = torch.sum(k[CC['PixelIdxList'][ii - 1]])
            if currsum < 0.1:
                k[CC['PixelIdxList'][ii - 1]] = 0
        
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
    
    return k, lambda_dark, lambda_grad, S

# You'll need to implement or import the missing functions (e.g., L0Deblur_dark_chanel, L0Restoration, estimate_psf,
# threshold_pxpy_v1, and connected_components) as well as set up the visualization and image saving part.
