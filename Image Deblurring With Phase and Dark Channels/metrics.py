import cv2
import torch
from math import log10
from skimage.metrics import structural_similarity as s_sim

def psnr(path1, path2):
    image1 = cv2.imread(path1)  
    image2 = cv2.imread(path2)  

    image1_tensor = torch.from_numpy(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.from_numpy(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    mse = torch.mean((image1_tensor - image2_tensor) ** 2)

    max_pixel_value = 1.0  # Assuming normalized tensors in the range [0, 1], replace with 255 if needed

    if mse == 0:
        psnr = 100  # PSNR is infinite if images are identical
    else:
        psnr = 20.0 * log10(max_pixel_value) - 10.0 * log10(mse)

    return psnr

def ssim(path1, path2):   
    image1 = cv2.imread(path1)  
    image2 = cv2.imread(path2)
    # image2 = image2[50:-50,50:-50]

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # print(gray_image1.shape)
    # print(gray_image2.shape)
    ssim_index = s_sim(gray_image1, gray_image2)
    
    return ssim_index

