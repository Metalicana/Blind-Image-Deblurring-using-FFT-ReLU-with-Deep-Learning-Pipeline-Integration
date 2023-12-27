import torch
import torch.fft as fft
import torch.nn.functional as F
from cho_code.wrap_boundary_liu import wrap_boundary_liu
from cho_code.opt_fft_size import opt_fft_size
from misc import psf2otf
import math

def otf2psf(input_tensor, sz):
    # input_tensor = fft.ifftn(input_tensor)

    # print(torch.abs(input_tensor))
    # Calculate the required shifts to bring the central element to (0, 0)
    center_shift = [-input_tensor.shape[0] // 2, -input_tensor.shape[1] // 2]

    # Circularly shift the input tensor to center the PSF at the origin
    shifted_tensor = torch.roll(input_tensor, center_shift, dims=(0, 1))

    # Crop the shifted tensor to match the desired size (sz)
    cropped_tensor = shifted_tensor[:sz[0], :sz[1]]

    # Compute the 2D IFFT of the cropped tensor

    # Take the real part to get the PSF, as it should be real

    return cropped_tensor

def shift_dc_to_top_left(input_tensor):
    # Get the dimensions of the input tensor
    h, w = input_tensor.shape[-2], input_tensor.shape[-1]

    # Calculate the required shifts to bring the DC component to (1, 1)
    shift = [h // 2, w // 2]

    # Circularly shift the input tensor to move the DC component to (1, 1)
    shifted_tensor = torch.roll(input_tensor, shift, dims=(-2, -1))

    return shifted_tensor
# psf = torch.arange(1.0,26.0)
# psf = psf.reshape(5,5)

# otf = psf2otf(psf, (5,5))
# print(torch.abs(otf))

# px = torch.zeros((1,2))
# px[0,0] = 1
# px[0,1] = -1
# print(px)
# otfpx = psf2otf(px,(5,5))
# print(torch.abs(otfpx))
# ifftx = fft.ifftshift(otfpx)
# ifftx = fft.ifftshift(otfpx)
# ifftx = fft.ifftn(ifftx)

# ifftx = torch.roll(ifftx,[3,2],dims=(0,1))
# print(torch.abs(ifftx))
# psfpx = fft.ifftn(otfpx)
# psfpx = fft.fftshift(psfpx)
# print(torch.abs(psfpx))

# xxx = otf2psf(otfpx,(5,5))
# print(torch.abs(xxx))

# A = torch.rand(50, 50)  # Replace with your actual data
# u = torch.rand(1, 10)   # Replace with your actual data

# # Compute the size of the result
# result_size = A.size()[0] - u.size()[1] + 1

# # Expand dimensions of u to match A
# u = u.view(1, 1, -1)

# # Perform valid column-wise convolutions using convolution
# result = torch.nn.functional.conv1d(A.view(1, 1, -1), u, padding=0)

# # Reshape the result to the desired shape
# result = result.view(result_size, -1)

def conv2(A, B, shape):
    #Pad A such that A has dimension (A.shape[0] + 2*B.shape[0] -2, A.shape[1] + 2*B.shape[1] - 2)
    #You do regular convolution.
    #then if it is valid, return just the center crop version of it
    #Expect A and B to be 2 dimensional matrieces
    #if shape is full then return full
    padCol = int((B.shape[1]*2 - 2)/2)
    padRow = int((B.shape[0]*2 - 2)/2)
    cropX, cropY = (A.shape[0]-B.shape[0]+1, A.shape[1]-B.shape[1]+1)
    
    padding = (padCol, padCol, padRow, padRow)
    A = F.pad(A,padding, value=0)
    
    B = torch.flip(B,[0,1])
    res = F.conv2d(A.unsqueeze(0).unsqueeze(0),B.unsqueeze(0).unsqueeze(0))
    res = res.squeeze() 
    if shape=='full':
        return res
    else:
        M,N = res.shape
        row = (M - cropX)//2
        col = (N - cropY)//2
        return res[row:row+cropX, col:col+cropY] 
    
latent = [[0.8147, 0.0975, 0.1576, 0.1419, 0.6557, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712],
          [0.9058, 0.2785, 0.9706, 0.4218, 0.0357, 0.7431, 0.3922, 0.6555, 0.1712, 0.7060],
          [0.1270, 0.5469, 0.9572, 0.9157, 0.8491, 0.3922, 0.6555, 0.1712, 0.7060, 0.0318],
          [0.9134, 0.9575, 0.4854, 0.7922, 0.9340, 0.6555, 0.1712, 0.7060, 0.0318, 0.2769],
          [0.6324, 0.9649, 0.8003, 0.9595, 0.6787, 0.1712, 0.7060, 0.0318, 0.2769, 0.0462],
          [0.0975, 0.1576, 0.1419, 0.6557, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712, 0.7060],
          [0.2785, 0.9706, 0.4218, 0.0357, 0.7431, 0.3922, 0.6555, 0.1712, 0.7060, 0.0318],
          [0.5469, 0.9572, 0.9157, 0.8491, 0.3922, 0.6555, 0.1712, 0.7060, 0.0318, 0.2769],
          [0.9575, 0.4854, 0.7922, 0.9340, 0.6555, 0.1712, 0.7060, 0.0318, 0.2769, 0.0462],
          [0.9649, 0.8003, 0.9595, 0.6787, 0.1712, 0.7060, 0.0318, 0.2769, 0.0462, 0.0971]]

# Convert the list to a PyTorch tensor
latent_tensor = torch.tensor(latent)
from misc import fft2
from misc import ifft2
from misc import otf2psf

hogarBal = psf2otf(latent_tensor, [15,15])
# print(hogarBal)
putkirBal = otf2psf(hogarBal, [10,10])
# print(torch.abs(putkirBal))
# haha = ifft2(hogarBal)
x = torch.tensor([[1,-1]])
y = psf2otf(x, [5,5])
# print(y)
z = otf2psf(y,[1,2])
# print(z)

# print(torch.abs(haha))

# shiftRight, shiftBottom = int((haha.shape[1]+2)/2) , int((haha.shape[0]+2)/2)
# centerX, centerY = shiftBottom, shiftRight
# shiftRight-=1
# shiftBottom-=1

# centerX -= 1
# centerY -= 1

# cray = torch.roll(haha, (shiftRight, shiftBottom), dims=[1,0])
# sz = [3, 1]
# top = int(sz[0]/2)
# bottom = sz[0] - top - 1
# left =int(sz[1] / 2)
# right = sz[1] - left - 1

# print(torch.abs(cray[centerX-top:centerX+bottom+1, centerY-left:centerY+right+1]))

# print(torch.abs(cray))

# print(psf2otf(latent_tensor, [13,13]))
# padRight = 3
# padBottom = 3
# padding = (0,3,0,3)
# leftShift, topShift = int((latent_tensor.shape[1]+2) / 2), int((latent_tensor.shape[0]+2) / 2)
# leftShift -= 1
# topShift -= 1
# latent_tensor = F.pad(latent_tensor,padding, value=0)


# print(leftShift)
# print(topShift)
# print(latent_tensor.shape)
# patent = torch.roll(latent_tensor,shifts=(-leftShift, -topShift), dims=[1,0])
# print(patent)
# patent = patent.t()
# patent = torch.fft.fft(patent).t()
# patent = torch.fft.fft(patent)
# print(patent)

# pxpy = torch.tensor([[1],[-1]])
# print(pxpy.shape)

# fuckyou = F.pad(pxpy, padding, value=0)
# print(fuckyou)
# dx = torch.tensor([[-1, 1],[0,0]]).type(torch.float32)
# dy = torch.tensor([[-1, 0],[1, 0]]).type(torch.float32)


# px = conv2(latent_tensor, dx, 'valid')
# py = conv2(latent_tensor, dy, 'valid')


# pm = px**2 + py**2

# pd = torch.atan(py / px)
# print(pd)
# import math 
# cond1 = (pd >= 0) & (pd < (math.pi / 4.0))

# result = pm[cond1]
# print(result)

# histogram = torch.histc(result, 33334, 0, 2.00006).unsqueeze(1)
# histogram = torch.flip(histogram, [0])
# histogram = torch.cumsum(histogram,dim=0)
# print(histogram)

# def threshold_pxpy_v1(latent, psf_size, threshold=None):
#     # Check if threshold is provided
#     if threshold is None:
#         threshold = 0
#         b_estimate_threshold = True
#     else:
#         b_estimate_threshold = False

#     # Denoised is the same as latent in this version
#     denoised = latent.clone()

#     # Define derivative filters
#     dx = torch.tensor([[-1, 1], [0, 0]], dtype=torch.float)
#     dy = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float)

#     # Compute gradients
#     px = conv2(denoised, dx, 'valid')
#     py = conv2(denoised, dy, 'valid')
#     pm = px**2 + py**2

#     # Estimate threshold if necessary
#     if b_estimate_threshold:
#         pd = torch.atan(py/ px)
#         pm_steps = torch.arange(0, 2, 0.00006)

#         pm_pos1 = pm[(pd >= 0) & (pd < (math.pi / 4.0))]
#         pm_pos2 = pm[(pd >= (math.pi / 4.0)) & (pd < (math.pi / 2.0))]
#         pm_neg1 = pm[(pd >= (-math.pi / 4.0)) & (pd < 0)]
#         pm_neg2 = pm[(pd >= (-math.pi / 2.0)) & (pd < (-math.pi / 4.0))]

#         H1 = torch.cumsum(torch.flip(torch.histc(pm_pos1, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
#         H2 = torch.cumsum(torch.flip(torch.histc(pm_pos2, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
#         H3 = torch.cumsum(torch.flip(torch.histc(pm_neg1,pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)
#         H4 = torch.cumsum(torch.flip(torch.histc(pm_neg2, pm_steps.shape[0],0,2.00006).unsqueeze(1), [0]), 0)

#         th = psf_size.max() * 20  # Adjust this as needed
#         if th <= 10:
#             th = 10
#         th = 5
#         for t in range(len(pm_steps)):
#             min_h = min(H1[t], H2[t], H3[t], H4[t])
#             # print(min_h)
#             if min_h >= th:
#                 threshold = pm_steps[-t]
#                 break

#     # Thresholding
#     print(threshold)
#     m = pm < threshold
#     while torch.all(m == 1):
#         threshold = threshold * 0.81
#         m = pm < threshold
#     px[m] = 0
#     py[m] = 0

#     # Modify threshold based on the condition
    
#     if b_estimate_threshold:
#         threshold = threshold
#     else:
#         threshold = threshold / 1.1

#     return px, py, threshold

# ks = torch.tensor([13,7])
# px, py, threshold = threshold_pxpy_v1(latent_tensor,ks )


# print(threshold)
# from skimage import measure
# grid = [[0, 0, 0, 0, 0, 0, 0],
#         [0, 0.0144, 0.0530, 0.0627, 0.0373, 0, 0],
#         [0.0402, 0.1092, 0.1773, 0.1757, 0.1035, 0.0344, 0.0125],
#         [0.0165, 0.0392, 0.0398, 0.0386, 0.0257, 0.0102, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0.0098, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0]]

# # Convert the grid to a PyTorch tensor
# tensor = torch.as_tensor(grid)
# t2 = tensor.clone()
# print(tensor)
# binary_image = tensor.clone()
# binary_image = binary_image.numpy()
# binary_image[binary_image>0] = 1
# # print (measure.label(binary_image, background=0.,connectivity=1))
# t = measure.label(binary_image)

# #print(tensor)
# t = torch.from_numpy(t)
# num = torch.max(t)
# threshold = 0.1
# #print("")
# for n in range(1, num+1):
#     indices = torch.nonzero(t == n).tolist()
#     sum = tensor[torch.tensor(indices)[:, 0], torch.tensor(indices)[:, 1]].sum()
#     if sum<threshold:
#         for index in indices:
#             tensor[index[0], index[1]] = 0

# #print(tensor)
# def connected_components(bw):
#     CC = {}
#     t = bw.clone()
#     t = t.numpy()
#     t[t>0] = 1
#     lbl = measure.label(t)
#     lbl = torch.from_numpy(lbl)
#     x = torch.max(lbl)
#     CC['NumObjects'] = x
#     CC['PixelIdxList'] = []
#     num = CC['NumObjects']
#     for n in range(1, num+1):
#         indices = torch.nonzero(lbl == n).tolist()
#         CC['PixelIdxList'].append(indices)

#     return CC

# print(connected_components(t2))
# properties = measure.regionprops(labeled_image)
# print(binary_image)
# # Print the tensor
# print(labeled_image)

# print(properties)

#Image to tensor
# import cv2
# from PIL import Image
# import torchvision.transforms as transforms

# image_path = 'images/post_blur.png'
# ipt = Image.open(image_path)

# def process_image(input_image) -> torch.Tensor:
#   transform = transforms.Compose([
#       transforms.PILToTensor()
#   ])
#   input_tensor = transform(input_image).type(torch.float32)
#   return input_tensor


# image = process_image(ipt)
# image = image.permute(1,2,0)

# true_gray = image[:,:,0]*0.2989+ image[:,:,1]*0.587 + image[:,:,2]*0.114
# true_gray = torch.round(true_gray)
# true_gray = true_gray / 255.0
# print(true_gray[0:5,0:5])
# import numpy as np
# mat = torch.arange(1,120*120 + 1).reshape(120,120).type(torch.float32)
# mat /= (120*120)


# kernel_np = np.eye(7, dtype=np.float32)

# from misc import conv2



# # Convert the NumPy array to a PyTorch tensor
# tensor_kernel = torch.tensor(kernel_np)
# print(conv2(mat.unsqueeze(2), tensor_kernel, 'valid').squeeze())

# from L0Restoration import L0Restoration

# S = L0Restoration(mat.unsqueeze(2), tensor_kernel, .1415, 2.0)

#Testing Estimate PSF
# blurred_x = torch.arange(1,120*120 + 1).reshape(120,120).type(torch.float32)
# blurred_x /= (120.0*120.0)
# print(blurred_x[0:5,0:5])

# blurred_y = torch.arange(120*120+1, 2*120*120 + 1).reshape(120,120).type(torch.float32)
# blurred_y /= (2* 120.0*120.0)
# print(blurred_y[0:5,0:5])

# latent_x = torch.arange(2*120*120+1, 3*120*120 + 1).reshape(120,120).type(torch.float32)
# latent_x /= (3* 120.0*120.0)
# print(latent_x[0:5,0:5])

# latent_y = torch.arange(3*120*120+1, 4*120*120 + 1).reshape(120,120).type(torch.float32)
# latent_y /= (4* 120.0*120.0)
# print(latent_y[0:5,0:5])

# from estimate_psf import estimate_psf

# print(estimate_psf(blurred_x, blurred_y, latent_x, latent_y, 2, [9,9]))
# import numpy as np

# opts = {
#         'prescale': 1,   # Downsampling
#         'xk_iter': 5,    # Iterations
#         'gamma_correct': 1.0,
#         'k_thresh': 20,
#         'kernel_size':25,
#     }

# lambda_dark = 4e-3
# #Experimenting with lambda_dark set to 0
# lambda_dark = 0
# lambda_grad = 4e-3
# lambda_tv = 0.003
# lambda_l0 = 5e-4
# weight_ring = 1

# threshold = 0.0086

# kernel_np = np.eye(7, dtype=np.float32)
# tensor_kernel = torch.tensor(kernel_np)

# blurred_x = torch.arange(1,120*120 + 1).reshape(120,120).type(torch.float32)
# blurred_x /= (120.0*120.0)
# from blind_deconv_main import blind_deconv_main
# from blind_deconv import blind_deconv
# # k , lambda_dark, lambda_grad, S = blind_deconv_main(blurred_x.unsqueeze(2)  , tensor_kernel, lambda_dark,lambda_grad,threshold,opts)
# kernel, interim = blind_deconv(blurred_x, lambda_dark, lambda_grad, opts)
# print(kernel)

#Testing for Adjust psf center 

# #Create a 9x9 tensor representing a kernel similar to the comment below
# [         0,         0,         0,         0,         0,         0,         0;
#     0.0074,    0.0274,    0.0366,    0.0480,    0.0385,    0.0201,    0;
#     0.0771,    0.1128,    0.1100,    0.1057,    0.0996,    0.0950,    0.0451;
#     0.0286,    0.0395,    0.0219,    0.0148,    0.0265,    0.0454,         0;
#          0,         0,         0,         0,         0,         0,         0;
#          0,         0,         0,         0,         0,         0,         0;
#          0,         0,         0,         0,         0,         0,         0];
newPsf = torch.tensor([[0,0,0,0,0,0,0],
                        [0.0074,0.0274,0.0366,0.0480,0.0385,0.0201,0],
                        [0.0771,0.1128,0.1100,0.1057,0.0996,0.0950,0.0451],
                        [0.0286,0.0395,0.0219,0.0148,0.0265,0.0454,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0]])

print(newPsf)
print(newPsf.shape)
from cho_code.adjust_psf_center import adjust_psf_center
newPsf = adjust_psf_center(newPsf)
print(newPsf)
