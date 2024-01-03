import torch
import torch.nn.functional as F
import math
from cho_code.adjust_psf_center import adjust_psf_center
from cho_code.threshold_pxpy_v1 import threshold_pxpy_v1
from blind_deconv_main import blind_deconv_main
from misc import conv2
from misc import conv2Vector
def blind_deconv(y, lambda_ftr,lambda_dark, lambda_grad, opts):
    if opts['gamma_correct'] != 1:
        y = y.pow(opts['gamma_correct'])
    
    b = torch.zeros(opts['kernel_size'])
    
    ret = math.sqrt(0.5)
    maxitr = max(math.floor(math.log(5 / opts['kernel_size']) / math.log(ret)), 0)
    num_scales = maxitr + 1
    print(f"Maximum iteration level is {num_scales}")

    retv = ret ** torch.arange(maxitr + 1)
    k1list = torch.ceil(opts['kernel_size'] * retv).to(torch.int)  
    k1list += (k1list % 2 == 0).to(torch.int)
    k2list = k1list.clone()

    dx = torch.tensor([[-1, 1], [0, 0]], dtype=torch.float32)
    dy = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float32)

    ks = None
    k1 = None
    k2 = None

    for s in range(num_scales, 0, -1):
        if s == num_scales:
            ks = init_kernel(k1list[s - 1]).to(torch.float32)
            # print(ks)
            k1 = k1list[s - 1].to(torch.int)
            k2 = k1.to(torch.int)
           

        else:
            k1 = k1list[s - 1].to(torch.int)
            k2 = k1.to(torch.int)
            
            ks = resize_kernel(ks, 1 / ret, k1list[s - 1], k2list[s - 1])
            # print(ks)


        cret = retv[s - 1]
        # print(y[0:5,0:5])
        ys = downsample_image(y, cret)
        # print(ys[0:10,0:10])
        # ys = ys.permute(1,2,0)
        print(f"Processing scale {s}/{num_scales}; kernel size {k1}x{k2}; image size {ys.shape[0]}x{ys.shape[1]}")
        
        if s == num_scales:
            _, _, threshold = threshold_pxpy_v1(ys, torch.tensor(max(k1, k2)) )
        # print(threshold)
        # print(f'shape of input before going into blind_deconv_main {ys.unsqueeze(2).shape}')
        # print(f'Ks klooked like this ')
        
        ks, lambda_ftr, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(ys.unsqueeze(2), ks,lambda_ftr, lambda_dark, lambda_grad, threshold, opts)
        # print(ks)
        #remember to fix this later on 
        # print(ks)
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0
        sumk = ks.sum()
        ks /= sumk
        
        if s == 1:
            kernel = ks.clone()
            if opts['k_thresh'] > 0:
                kernel[kernel < kernel.max() / opts['k_thresh'] ] = 0
            else:
                kernel[kernel < 0] = 0
            kernel /= kernel.sum()
    
    return kernel, interim_latent

def init_kernel(minsize):
    k = torch.zeros(minsize, minsize)
    k[(minsize - 1) // 2 - 1, (minsize - 1) // 2 -1:(minsize - 1) // 2 + 1] = 1 / 2
    return k

def downsample_image(I, ret):
    if ret == 1:
        return I
    sig = 1 / math.pi * ret
    
    g0 = torch.arange(-50, 51) * 2 * math.pi

    sf = torch.exp(-0.5 * g0**2 * sig**2)

    sf /= sf.sum()
    csf = torch.cumsum(sf, dim=0)
    csf = torch.min(csf, csf.flip(0))

    ii = torch.where(csf > 0.05)[0]
    # print(ii)
    sf = sf[ii]
    sf = sf.unsqueeze(0)

    h,w = I.shape
    # print(I[:,434:449])
    I = conv2Vector(sf,sf,I,'valid')
    # print(I[0:10,0:10])
    gx, gy = torch.meshgrid(torch.arange(1, I.shape[0] + 1, step=1 / ret), torch.arange(1, I.shape[1] + 1, step=1 / ret))
    # print(gx)
  
    gx = gx.squeeze()
    
    gxMax = gx.max()
    gxMin = gx.min()
    gx = (2*(gx-gxMin)/(gxMax-gxMin))-1
    # print(gx[0:5,0:5])
    gy = gy.squeeze()
    # print(gy[0:5,0:5])
    gyMax = gy.max()
    gyMin = gy.min()
    gy = (2*(gy-gyMin)/(gyMax-gyMin))-1
    # return I
    I = I.view(1,1,I.shape[0], I.shape[1])
    grid = torch.stack((gy, gx), dim=-1)
    grid = grid.unsqueeze(0)

    # print('Grid format:')
    # print(grid.shape)
    # print(grid)
    
    sI = F.grid_sample(I, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # print(sI.squeeze()[0:10,0:10])
    
    # print(f'SI is {sI}')
    return sI.squeeze()

def resize_kernel(k, ret, k1, k2):
    k = F.interpolate(k.unsqueeze(0).unsqueeze(0), scale_factor=ret, mode='bilinear', align_corners=False)
    k = k.squeeze().clamp(0)
    k = fixsize(k, k1, k2)
    
    if k.max() > 0:
        k /= k.sum()
    
    return k

def fixsize(f, nk1, nk2):
    k1, k2 = f.shape

    while k1 != nk1 or k2 != nk2:
        if k1 > nk1:
            s = f.sum(dim=1)
            if s[0] < s[-1]:
                f = f[1:]
            else:
                f = f[:-1]
        
        if k1 < nk1:
            s = f.sum(dim=1)
            if s[0] < s[-1]:
                tf = torch.zeros((k1 + 1, f.shape[1]))
                tf[:k1] = f
                f = tf
            else:
                tf = torch.zeros((k1 + 1, f.shape[1]))
                tf[1:] = f
                f = tf
        
        if k2 > nk2:
            s = f.sum(dim=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]
        
        if k2 < nk2:
            s = f.sum(dim=0)
            if s[0] < s[-1]:
                tf = torch.zeros((f.shape[0], k2 + 1))
                tf[:, :k2] = f
                f = tf
            else:
                tf = torch.zeros((f.shape[0], k2 + 1))
                tf[:, 1:] = f
                f = tf
        
        k1, k2 = f.shape
    
    return f

